from typing import Annotated, List, TypedDict, Literal
from enum import Enum
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import boto3
from langchain_aws import ChatBedrockConverse
from langchain.output_parsers import PydanticOutputParser
import requests

# LLM 초기화
def get_bedrock_model(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0", temperature=0):
    # Bedrock 클라이언트 생성
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"  # 사용 중인 리전으로 변경
    )
    
    # LangChain BedrockChat 모델 생성
    model = ChatBedrockConverse(
        client=bedrock_runtime,
        model_id=model_id,
        # model_kwargs={"temperature": temperature}
    )
    
    return model

llm = get_bedrock_model()

from mock_data import MOCK_DATA

# 외부 API 도구 정의
@tool
def get_merchant_data(merchant_id: str) -> dict:
    """가맹점 ID를 기반으로 가맹점 정보를 조회합니다.
    
    Args:
        merchant_id: 조회할 가맹점의 ID
    """
    if merchant_id in MOCK_DATA["merchants"]:
        return MOCK_DATA["merchants"][merchant_id]
    else:
        return {"error": f"가맹점 ID {merchant_id}에 대한 정보가 없습니다."}
    
    # 실제 구현에서는 실제 API 호출
    # response = requests.get(f"https://api.merchant-db.com/merchants/{merchant_id}")
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return {"error": f"가맹점 정보를 찾을 수 없습니다. 상태 코드: {response.status_code}"}

@tool
def get_merchant_risk_score(merchant_id: str) -> dict:
    """가맹점의 위험도 점수를 조회합니다.
    
    Args:
        merchant_id: 조회할 가맹점의 ID
    """
    if merchant_id in MOCK_DATA["merchant_risk"]:
        return MOCK_DATA["merchant_risk"][merchant_id]
    else:
        return {"error": f"가맹점 ID {merchant_id}에 대한 위험도 정보가 없습니다."}
    
    # response = requests.get(f"https://api.merchant-db.com/merchants/{merchant_id}/risk")
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return {"error": f"가맹점 위험도 정보를 찾을 수 없습니다."}

@tool
def get_user_profile(user_id: str) -> dict:
    """사용자 ID를 기반으로 프로필 정보를 조회합니다.
    
    Args:
        user_id: 조회할 사용자의 ID
    """
    if user_id in MOCK_DATA["users"]:
        return MOCK_DATA["users"][user_id]
    else:
        return {"error": f"사용자 ID {user_id}에 대한 정보가 없습니다."}

    # response = requests.get(f"https://api.user-db.com/users/{user_id}")
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return {"error": f"사용자 정보를 찾을 수 없습니다."}

@tool
def get_user_transaction_history(user_id: str, days: int = 90) -> dict:
    """사용자의 거래 내역을 조회합니다.
    
    Args:
        user_id: 조회할 사용자의 ID
        days: 조회할 기간(일)
    """
    if user_id in MOCK_DATA["transactions"]:
        return {
            "user_id": user_id,
            "transactions": MOCK_DATA["transactions"][user_id],
            "period_days": days
        }
    else:
        return {"error": f"사용자 ID {user_id}에 대한 거래 내역이 없습니다."}
    # response = requests.get(
    #     f"https://api.user-db.com/users/{user_id}/transactions",
    #     params={"days": days}
    # )
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return {"error": f"사용자 거래 내역을 찾을 수 없습니다."}

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnalysisResult(BaseModel):
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    # recommended_action: Optional[str] = None

# 상태 정의
class FraudDetectionState(TypedDict):
    transaction_data: dict  # 분석할 거래 데이터
    analysis_results: Annotated[list, operator.add]  # 각 Worker의 분석 결과
    final_report: dict  # 최종 탐지 보고서

# Worker 상태 정의
class WorkerState(TypedDict):
    transaction_data: dict
    analysis_results: Annotated[list, operator.add]

# 도구 바인딩
tools = [get_merchant_data, get_merchant_risk_score, get_user_profile, get_user_transaction_history]
tools_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
# merchant_tools = []
# merchant_tools_dict = {tool.name: tool for tool in merchant_tools}
# merchant_llm = llm.bind_tools(merchant_tools)

# user_tools = []
# user_tools_dict = {tool.name: tool for tool in user_tools}
# user_llm = llm.bind_tools(user_tools)

parser = PydanticOutputParser(pydantic_object=AnalysisResult)

# 1. Transaction Worker - 거래 자체의 특성 분석
def transaction_worker(state: WorkerState):
    """거래 자체의 패턴과 특성을 분석"""
    
    transaction_data = state["transaction_data"]

    transaction_agent_prompt = f"""당신은 거래 패턴 분석 전문가입니다. 현재 거래 자체의 특성을 분석하여 위험 거래 가능성을 평가하세요.
    
    다음을 고려하세요:
    - 거래 금액이 비정상적으로 크거나 작은지
    - 거래 시간(심야, 이른 새벽 등)이 의심스러운지
    - 거래 위치가 위험 지역으로 알려진 곳인지
    - 거래 유형과 결제 방법이 위험한지 (예: 익명성이 높은 결제수단)
    - 가맹점 유형이 위험 거래에 취약한 분야인지 (온라인 도박, 가상화폐 등)
    - 거래의 IP 주소나 디바이스 정보가 의심스러운지
    
    분석 결과를 아래 형식의 JSON으로 정확히 제공하세요:
    ```json
    {{
      "risk_level": "low", // "low", "medium", "high", "critical" 중 하나
      "confidence": 0.85, // 0.0에서 1.0 사이의 신뢰도 점수
      "reasoning": "분석 결과에 대한 상세 설명을 하나의 문자열로 작성하세요. 여러 이유가 있으면 문장으로 이어서 작성하세요.",
      "recommended_action": "권장 조치를 하나의 문자열로 작성하세요. 여러 조치가 필요하면 문장으로 이어서 작성하세요."
    }}
    ```
    
    중요: reasoning과 recommended_action은 반드시 문자열로 작성해야 합니다. 배열이나 리스트 형태가 아니라 단일 문자열이어야 합니다.
    JSON 형식을 정확히 지켜주세요. 다른 텍스트나 설명은 포함하지 마세요."""
    
    # 거래 데이터 분석
    analysis = llm.invoke([
        SystemMessage(content=transaction_agent_prompt),
        HumanMessage(content=f"거래 정보: {transaction_data}")
    ])
    
    # print("#### transaction analysis result ####")
    # print(parser.parse(analysis.content))
    
    return {
        "analysis_results": [{
            "component": "transaction_analysis",
            "analysis": parser.parse(analysis.content)
        }]
    }

# 2. Merchant Worker - 가맹점 정보 활용 (Agent 패턴)
def merchant_worker(state: WorkerState):
    """가맹점 정보를 조회하고 분석"""
    
    transaction_data = state["transaction_data"]
    merchant_id = transaction_data.get("merchant_id", "unknown")
    
    merchant_agent_prompt = f"""당신은 가맹점 분석 전문가입니다. 필요한 가맹점 정보를 조회하고 해당 거래의 위험성을 분석하세요
    
    다음을 고려하세요:
    - 가맹점이 알려진 위험 거래 사례와 연결되어 있는지
    - IP 주소나 기기 ID가 의심스러운 활동과 연관되어 있는지
    - 비정상적인 자금 흐름 패턴이 있는지
    - 사용자의 정보는 분석하지 마시오
    
    분석 결과를 아래 형식의 JSON으로 정확히 제공하세요:
    ```json
    {{
      "risk_level": "low", // "low", "medium", "high", "critical" 중 하나
      "confidence": 0.85, // 0.0에서 1.0 사이의 신뢰도 점수
      "reasoning": "분석 결과에 대한 상세 설명을 하나의 문자열로 작성하세요. 여러 이유가 있으면 문장으로 이어서 작성하세요.",
      "recommended_action": "권장 조치를 하나의 문자열로 작성하세요. 여러 조치가 필요하면 문장으로 이어서 작성하세요."
    }}
    ```
    
    중요: reasoning과 recommended_action은 반드시 문자열로 작성해야 합니다. 배열이나 리스트 형태가 아니라 단일 문자열이어야 합니다.
    JSON 형식을 정확히 지켜주세요. 다른 텍스트나 설명은 포함하지 마세요.

    중요: 분석에 꼭 필요한 도구만 사용하세요. 모든 도구를 사용할 필요는 없습니다.
    불필요한 API 호출은 비용과 시간을 낭비합니다.
    """
    
    # 대화 초기화
    messages = [
        SystemMessage(content=merchant_agent_prompt),
        HumanMessage(content=f"merchant_id: {merchant_id}인 가맹점에서 발생한 다음 거래를 분석해주세요. 필요한 정보는 도구를 사용하여 조회하고, 불필요한 도구는 사용하지 마세요:\n\n{transaction_data}")
    ]
    
    
    # 에이전트 루프
    while True:
        # LLM 호출
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # 도구 호출이 있는 경우
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool = tools_dict[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
                tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
            # 도구 결과 추가
            messages.extend(tool_results)
        else:
            # 최종 분석 완료
            break
    
    # print("### 가맹점 분석 ###")
    # print(parser.parse(messages[-1].content))
    # 최종 분석 결과 반환
    return {
        "analysis_results": [{
            "component": "merchant_analysis",
            "analysis": parser.parse(messages[-1].content)
        }]
    }

# 3. User Behavior Worker - 사용자 행동 분석 (Agent 패턴)
def user_behavior_worker(state: WorkerState):
    """사용자 행동 패턴 분석"""
    
    transaction_data = state["transaction_data"]
    user_id = transaction_data.get("user_id", "unknown")
    
    user_behavior_agent_prompt = f"""
    당신은 사용자 행동 분석 전문가입니다. 이 거래가 사용자의 일반적인 행동 패턴과 일치하는지 평가하세요.
    
    다음을 고려하세요:
    - 사용자의 과거 구매 이력과 현재 거래의 일관성
    - 새로운 기기나 위치에서의 접속 여부
    - 이전에 방문하지 않은 가맹점에서의 거래인지
    - 사용자의 일반적인 지출 패턴과의 부합성
    
    분석 결과를 아래 형식의 JSON으로 정확히 제공하세요:
    ```json
    {{
      "risk_level": "low", // "low", "medium", "high", "critical" 중 하나
      "confidence": 0.85, // 0.0에서 1.0 사이의 신뢰도 점수
      "reasoning": "분석 결과에 대한 상세 설명을 하나의 문자열로 작성하세요. 여러 이유가 있으면 문장으로 이어서 작성하세요.",
      "recommended_action": "권장 조치를 하나의 문자열로 작성하세요. 여러 조치가 필요하면 문장으로 이어서 작성하세요."
    }}
    ```
    
    중요: reasoning과 recommended_action은 반드시 문자열로 작성해야 합니다. 배열이나 리스트 형태가 아니라 단일 문자열이어야 합니다.
    JSON 형식을 정확히 지켜주세요. 다른 텍스트나 설명은 포함하지 마세요."""
    
    # 대화 초기화
    messages = [
        SystemMessage(content=user_behavior_agent_prompt),
        HumanMessage(content=f"user_id: {user_id}인 사용자의 다음 거래를 분석해주세요. 필요한 정보는 도구를 사용하여 조회하세요:\n\n{transaction_data}")
    ]
    
    # 에이전트 루프
    while True:
        # LLM 호출
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # 도구 호출이 있는 경우
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool = tools_dict[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
                tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
            # 도구 결과 추가
            messages.extend(tool_results)
        else:
            # 최종 분석 완료
            break
    
    # print("### 사용자 행동 분석 ###")
    # print(parser.parse(messages[-1].content))
    
    # 최종 분석 결과 반환
    return {
        "analysis_results": [{
            "component": "user_behavior_analysis",
            "analysis": parser.parse(messages[-1].content)
        }]
    }

# 결과 합성기 - 모든 분석 결과를 종합
def synthesizer(state: FraudDetectionState):
    """모든 분석 결과를 종합하여 최종 판단"""
    
    transaction_data = state["transaction_data"]
    analysis_results = state["analysis_results"]
    
    # 분석 결과 형식화
    formatted_results = ""
    for result in analysis_results:
        formatted_results += f"\n## {result['component'].upper()}\n{result['analysis']}"
    
    # 최종 판단
    final_report = llm.invoke([
        SystemMessage(content="당신은 위험 거래 탐지 전문가입니다. 여러 분석 결과를 종합하여 최종 위험 거래 여부를 판단하세요."),
        HumanMessage(content=f"다음은 거래 ID {transaction_data.get('transaction_id', 'unknown')}에 대한 다양한 분석 결과입니다.\n\n"
                     f"분석 결과:{formatted_results}\n\n"
                     f"이 거래가 위험 거래일 가능성, 신뢰도 점수(0-100), 판단 근거를 포함한 종합 보고서를 작성해주세요.")
    ])
    
    return {
        "final_report": {
            "transaction_id": transaction_data.get("transaction_id", "unknown"),
            "report": final_report.content
        }
    }

# Orchestrator - 작업 계획 및 할당
def orchestrator(state: FraudDetectionState):
    """거래를 분석하여 필요한 Worker 할당"""
    
    # 이 예제에서는 항상 모든 Worker를 실행
    # 실제 구현에서는 거래 특성에 따라 필요한 Worker만 선택적으로 실행 가능
    return {}

# Worker 할당 함수
def assign_workers(state: FraudDetectionState):
    """각 Worker에게 작업 할당"""
    return [
        Send("transaction_worker", {"transaction_data": state["transaction_data"]}),
        Send("merchant_worker", {"transaction_data": state["transaction_data"]}),
        Send("user_behavior_worker", {"transaction_data": state["transaction_data"]})
    ]

# 워크플로우 구축
workflow_builder = StateGraph(FraudDetectionState)

# 노드 추가
workflow_builder.add_node("orchestrator", orchestrator)
workflow_builder.add_node("transaction_worker", transaction_worker)
workflow_builder.add_node("merchant_worker", merchant_worker)
workflow_builder.add_node("user_behavior_worker", user_behavior_worker)
workflow_builder.add_node("synthesizer", synthesizer)

# 엣지 연결
workflow_builder.add_edge(START, "orchestrator")
workflow_builder.add_conditional_edges("orchestrator", assign_workers)
workflow_builder.add_edge("transaction_worker", "synthesizer")
workflow_builder.add_edge("merchant_worker", "synthesizer")
workflow_builder.add_edge("user_behavior_worker", "synthesizer")
workflow_builder.add_edge("synthesizer", END)

# 워크플로우 컴파일
fraud_detection_workflow = workflow_builder.compile()


        # "TX12345678": ,
# 테스트 실행 예시
sample_transaction = {
    "transaction_id": "TX98765432",
    "user_id": "U12345678",
    "merchant_id": "M87654321",
    "amount": 2500.00,
    "timestamp": "2023-09-15T02:30:45Z",
    "location": "Seoul, South Korea",
    "payment_method": "credit_card",
    "card_last_four": "4567",
    "device_id": "D-MOBILE-XYZ",
    "ip_address": "203.0.113.42",
    "transaction_type": "online_purchase"
}

sample_transaction = {
    "transaction_id": "TX12345678",
    "user_id": "U87654321",
    "merchant_id": "M12345678",
    "amount": 4800.00,
    "timestamp": "2023-09-14T22:15:30Z",
    "location": "Unknown",
    "payment_method": "credit_card",
    "card_last_four": "9876",
    "device_id": "D-UNKNOWN-ABC",
    "ip_address": "198.51.100.78",
    "transaction_type": "cash_advance"
}
# 워크플로우 실행
# fraud_detection_workflow.get_graph().draw_mermaid_png(output_file_path="graph.png")
result = fraud_detection_workflow.invoke({"transaction_data": sample_transaction})

# 결과 출력
print(result["final_report"]["report"])