import json
import random
import uuid
from datetime import datetime, timedelta

def generateHexId(length):
    return uuid.uuid4().hex[:length].upper()

def generateB2bDataset():
    totalUsers = 40
    dataset = []
    
    actionTypes = ["rfx_viewed", "bid_submitted"]
    companies = [f"COMPANY_{generateHexId(8)}" for _ in range(10)]

    for _ in range(totalUsers):
        userId = generateHexId(8)
        sessionId = f"sess_{generateHexId(8)}"
        
        # Rastgele bir baslangic zamani
        startBaseTime = datetime(2024, random.randint(1, 12), random.randint(1, 28), random.randint(8, 22), random.randint(0, 59))
        
        eventCount = random.randint(20, 30)
        sequentialEvents = []
        
        currentTime = startBaseTime
        
        hasBid = False
        hasRfx = False
        
        for seq in range(1, eventCount + 1):
            currentTime += timedelta(seconds=random.randint(10, 120))
            actionChoice = random.choice(actionTypes)
            
            eventBase = {
                "seq": seq,
                "event_type": "rfx" if actionChoice == "rfx_viewed" else "bid",
                "event_category": "procurement",
                "action_type": actionChoice,
                "platform": "web",
                "urgency_level": "low",
                "user_journey_stage": "retention",
                "is_automated": False,
                "created_date": currentTime.isoformat(timespec='microseconds') + "+03:00",
                "hour_of_day": currentTime.hour,
                "day_of_week": currentTime.strftime("%A"),
                "time_of_day": "evening" if currentTime.hour >= 18 else ("morning" if currentTime.hour < 12 else "afternoon"),
                "is_weekend": currentTime.weekday() >= 5
            }

            if actionChoice == "rfx_viewed":
                hasRfx = True
                eventBase.update({
                    "message_template": "RFX_VIEWED",
                    "message_normalized": "Teklif talebi görüntülendi.",
                    "tags": ["rfx", "viewed", "procurement", "supplier_action"],
                    "outcome": "neutral",
                    "sentiment": "neutral",
                    "actor_role": "supplier"
                })
            else:
                hasBid = True
                isDetailed = random.choice([True, False])
                buyerCompany = random.choice(companies)
                supplierCompany = random.choice(companies)
                
                if isDetailed:
                    eventBase.update({
                        "message_template": "BID_SUBMITTED_TO_BUYER",
                        "message_normalized": f"{supplierCompany} tarafından {buyerCompany} firmasının talebine teklif verildi.",
                        "tags": ["bid", "submitted", "procurement", "supplier_action", "has_buyer"],
                        "named_entities": {
                            "supplier_company": supplierCompany,
                            "buyer_company": buyerCompany
                        },
                        "outcome": "success",
                        "sentiment": "positive",
                        "actor_role": "supplier"
                    })
                else:
                    eventBase.update({
                        "message_template": "BID_SUBMITTED",
                        "message_normalized": "Tedarikçi teklif verdi.",
                        "tags": ["bid", "submitted", "procurement", "supplier_action"],
                        "outcome": "success",
                        "sentiment": "positive",
                        "actor_role": "supplier"
                    })
            
            sequentialEvents.append(eventBase)

        sessionDuration = (currentTime - startBaseTime).total_seconds() / 60.0

        userRecord = {
            "userId": userId,
            "sessionId": sessionId,
            "sessionStartTime": {
                "timestamp": startBaseTime.isoformat(timespec='microseconds') + "+03:00",
                "year": startBaseTime.year,
                "month": startBaseTime.month,
                "day_of_week": startBaseTime.strftime("%A"),
                "hour_of_day": startBaseTime.hour,
                "time_of_day": "evening" if startBaseTime.hour >= 18 else ("morning" if startBaseTime.hour < 12 else "afternoon"),
                "is_weekend": startBaseTime.weekday() >= 5,
                "platform": "web"
            },
            "summary": {
                "event_count": eventCount,
                "categories": ["procurement"],
                "action_types": list(set([e["action_type"] for e in sequentialEvents])),
                "unique_tags": list(set(tag for e in sequentialEvents for tag in e["tags"])),
                "has_failure": False,
                "has_purchase": False,
                "has_bid": hasBid,
                "has_rfx": hasRfx,
                "has_order": False,
                "session_end_time": currentTime.isoformat(timespec='microseconds') + "+03:00",
                "duration_minutes": round(sessionDuration, 1)
            },
            "sequentialEvents": sequentialEvents
        }
        
        dataset.append(userRecord)

    with open("B2bUserDataset.json", "w", encoding="utf-8") as fileHandle:
        json.dump(dataset, fileHandle, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generateB2bDataset()
    print("B2bUserDataset.json basariyla olusturuldu.")