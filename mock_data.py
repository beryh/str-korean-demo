# 한국 카드 거래 데이터
MOCK_DATA = {
    # 가맹점 기본 정보
    "merchants": {
        "M87654321": {
            "merchant_id": "M87654321",
            "name": "테크가제트몰",
            "category": "electronics",
            "establishment_date": "2015-03-12",
            "address": "서울시 디지털로 123",
            "avg_transaction_value": 350750,
            "currency": "KRW",
            "monthly_transaction_count": 2800,
            "status": "active"
        },
        "M12345678": {
            "merchant_id": "M12345678",
            "name": "퀵캐시ATM",
            "category": "financial_services",
            "establishment_date": "2020-11-05",
            "address": "부산시 머니거리 456",
            "avg_transaction_value": 120250,
            "currency": "KRW",
            "monthly_transaction_count": 5200,
            "status": "under_review"
        },
        "M98765432": {
            "merchant_id": "M98765432",
            "name": "글로벌샵온라인",
            "category": "general_merchandise",
            "establishment_date": "2018-07-22",
            "address": "서울시 웹 대로 789",
            "avg_transaction_value": 89500,
            "monthly_transaction_count": 12500,
            "status": "active"
        }
    },

    # 가맹점 위험도 점수
    "merchant_risk": {
        "M87654321": {
            "risk_score": 15,
            "risk_level": "low",
            "flags": ["none"],
            "last_assessment": "2023-08-10"
        },
        "M12345678": {
            "risk_score": 78,
            "risk_level": "high",
            "flags": ["unusual_activity", "recent_complaints"],
            "last_assessment": "2023-09-05"
        },
        "M98765432": {
            "risk_score": 32,
            "risk_level": "medium",
            "flags": ["rapid_growth"],
            "last_assessment": "2023-08-22"
        }
    },

    # 사용자 프로필 정보
    "users": {
        "U12345678": {
            "user_id": "U12345678",
            "name": "김민지",
            "age": 35,
            "account_creation_date": "2018-06-15",
            "credit_score": 725,
            "account_type": "premium",
            "occupation": "소프트웨어 엔지니어",
            "typical_login_locations": ["서울", "인천"],
            "typical_devices": ["iPhone", "MacBook"]
        },
        "U87654321": {
            "user_id": "U87654321",
            "name": "박지성",
            "age": 27,
            "account_creation_date": "2021-01-10",
            "credit_score": 680,
            "account_type": "standard",
            "occupation": "학생",
            "typical_login_locations": ["부산"],
            "typical_devices": ["Android Phone"]
        },
        "U13579246": {
            "user_id": "U13579246",
            "name": "이수진",
            "age": 42,
            "account_creation_date": "2016-11-20",
            "credit_score": 810,
            "account_type": "vip",
            "occupation": "의사",
            "typical_login_locations": ["서울", "제주"],
            "typical_devices": ["iPad", "Windows Laptop"]
        }
    },

    # 사용자 거래 내역
    "transactions": {
        "U12345678": [
            {
                "date": "2023-09-10",
                "amount": 54200,
                "currency": "KRW",
                "merchant": "로컬커피숍",
                "type": "in-person",
                "location": "서울"
            },
            {
                "date": "2023-09-08",
                "amount": 425750,
                "currency": "KRW",
                "merchant": "테크가제트몰",
                "type": "online",
                "location": "서울"
            },
            {
                "date": "2023-09-05",
                "amount": 82300,
                "currency": "KRW",
                "merchant": "서울식품점",
                "type": "in-person",
                "location": "서울"
            }
        ],
        "U87654321": [
            {
                "date": "2023-09-12",
                "amount": 1200000,
                "currency": "KRW",
                "merchant": "대학교도서관",
                "type": "in-person",
                "location": "부산"
            },
            {
                "date": "2023-09-10",
                "amount": 35400,
                "currency": "KRW",
                "merchant": "스트리밍서비스",
                "type": "subscription",
                "location": "online"
            },
            {
                "date": "2023-09-01",
                "amount": 2500000,
                "currency": "KRW",
                "merchant": "글로벌샵온라인",
                "type": "online",
                "location": "online"
            }
        ],
        "U13579246": [
            {
                "date": "2023-09-14",
                "amount": 320500,
                "currency": "KRW",
                "merchant": "럭셔리다이닝",
                "type": "in-person",
                "location": "서울"
            },
            {
                "date": "2023-09-10",
                "amount": 4500000,
                "currency": "KRW",
                "merchant": "의료장비",
                "type": "online",
                "location": "online"
            },
            {
                "date": "2023-09-02",
                "amount": 850000,
                "currency": "KRW",
                "merchant": "항공권",
                "type": "online",
                "location": "online"
            }
        ]
    },

    # 의심 거래 시나리오
    "suspicious_transactions": [
        {
            "transaction_id": "TX98765432",
            "user_id": "U12345678",
            "merchant_id": "M87654321",
            "amount": 2500000,
            "currency": "KRW",
            "timestamp": "2023-09-15T02:30:45Z",
            "location": "서울, 대한민국",
            "payment_method": "credit_card",
            "card_last_four": "4567",
            "device_id": "D-MOBILE-XYZ",
            "ip_address": "203.0.113.42",
            "transaction_type": "online_purchase"
        },
        {
            "transaction_id": "TX12345678",
            "user_id": "U87654321",
            "merchant_id": "M12345678",
            "amount": 480000,
            "currency": "KRW",
            "timestamp": "2023-09-14T22:15:30Z",
            "location": "Unknown",
            "payment_method": "credit_card",
            "card_last_four": "9876",
            "device_id": "D-UNKNOWN-ABC",
            "ip_address": "198.51.100.78",
            "transaction_type": "cash_advance"
        },
        {
            "transaction_id": "TX87654321",
            "user_id": "U13579246",
            "merchant_id": "M98765432",
            "amount": 350000,
            "currency": "KRW",
            "timestamp": "2023-09-15T10:45:12Z",
            "location": "제주, 대한민국",
            "payment_method": "debit_card",
            "card_last_four": "1357",
            "device_id": "D-TABLET-LMN",
            "ip_address": "203.0.113.100",
            "transaction_type": "online_purchase"
        }
    ]
}
