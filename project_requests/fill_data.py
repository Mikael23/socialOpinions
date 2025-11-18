import random
import requests

BASE_URL = "http://localhost:8080"  # change if needed

# ---------- API helpers ----------

def create_user(session, user):
    resp = session.post(f"{BASE_URL}/api/users/register", json=user)
    resp.raise_for_status()
    data = resp.json()
    print("Created user:", data)
    return data.get("id"), data.get("username", user["username"])

def create_area(session, area):
    resp = session.post(f"{BASE_URL}/api/areas", json=area)
    resp.raise_for_status()
    data = resp.json()
    print("Created area:", data)
    return data.get("id")

def create_survey(session, survey):
    resp = session.post(f"{BASE_URL}/api/surveys", json=survey)
    resp.raise_for_status()
    data = resp.json()
    print("Created survey:", data)
    return data.get("surveyId")

def create_survey_response(session, payload):
    resp = session.post(f"{BASE_URL}/api/survey-responses", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print("Created survey response:", data)
    return data.get("id")

# ---------- people-focused comments ----------

def generate_comment(mark: int, name: str) -> str:
    very_positive = [
        f"{name} is very reliable and always keeps promises.",
        f"Working with {name} is easy, calm and productive.",
        f"{name} listens carefully and respects other people's opinions.",
        f"{name} is organized, responsible and communicates clearly.",
    ]
    positive = [
        f"{name} is generally kind and supportive.",
        f"{name} usually finishes tasks on time and takes work seriously.",
        f"People feel comfortable sharing ideas with {name}.",
    ]
    neutral = [
        f"{name} has both strong and weak sides, like everyone.",
        f"Sometimes {name} is focused and calm, sometimes a bit distracted.",
        f"In general {name} is okay, but there is room to grow.",
    ]
    negative = [
        f"{name} often reacts emotionally and can be impatient.",
        f"Sometimes {name} does not listen carefully and interrupts others.",
        f"{name} can be inconsistent and forget about agreements.",
    ]
    very_negative = [
        f"{name} often creates tension in the group.",
        f"People sometimes feel uncomfortable sharing feedback with {name}.",
        f"{name} rarely takes responsibility for mistakes.",
    ]

    if mark == 5:
        base = random.choice(very_positive)
    elif mark == 4:
        base = random.choice(positive)
    elif mark == 3:
        base = random.choice(neutral)
    elif mark == 2:
        base = random.choice(negative)
    else:  # mark == 1
        base = random.choice(very_negative)

    extra = [
        "In general, this is the main impression.",
        "This behavior repeats quite often.",
        "This is visible both in work and in personal communication.",
        "Other people notice the same things.",
    ]
    return base + " " + random.choice(extra)

# ---------- main ----------

def main():
    session = requests.Session()

    # 1) Create 5 users
    users_payload = [
        {
            "username": "vasya",
            "email": "vasya@example.com",
            "password": "Password1!",
            "firstName": "Vasya",
            "lastName": "Pupkin",
            "dateOfBirth": "1990-01-01",
            "gender": "MALE",
            "phoneNumber": "+1111111111",
            "bio": "Poetry lover.",
            "profilePictureUrl": ""
        },
        {
            "username": "maria",
            "email": "maria@example.com",
            "password": "Password1!",
            "firstName": "Maria",
            "lastName": "Ivanova",
            "dateOfBirth": "1992-02-02",
            "gender": "FEMALE",
            "phoneNumber": "+2222222222",
            "bio": "Food reviewer.",
            "profilePictureUrl": ""
        },
        {
            "username": "alex",
            "email": "alex@example.com",
            "password": "Password1!",
            "firstName": "Alex",
            "lastName": "Smith",
            "dateOfBirth": "1988-03-03",
            "gender": "OTHER",
            "phoneNumber": "+3333333333",
            "bio": "Tech and gadgets.",
            "profilePictureUrl": ""
        },
        {
            "username": "olga",
            "email": "olga@example.com",
            "password": "Password1!",
            "firstName": "Olga",
            "lastName": "Petrova",
            "dateOfBirth": "1995-04-04",
            "gender": "FEMALE",
            "phoneNumber": "+4444444444",
            "bio": "Travel addict.",
            "profilePictureUrl": ""
        },
        {
            "username": "john",
            "email": "john@example.com",
            "password": "Password1!",
            "firstName": "John",
            "lastName": "Doe",
            "dateOfBirth": "1985-05-05",
            "gender": "MALE",
            "phoneNumber": "+5555555555",
            "bio": "General reviewer.",
            "profilePictureUrl": ""
        },
    ]

    users = []  # list of (userId, username)
    for u in users_payload:
        uid, uname = create_user(session, u)
        users.append((uid, uname))

    # 2) For each user: create 3 areas + 1 survey per area
    surveys = []  # list of (surveyId, subjectName)
    for user_id, username in users:
        for i in range(1, 4):
            area_req = {
                "ownerId": user_id,
                "title": f"Opinions about {username} #{i}",
                "description": f"Area {i} for collecting opinions about {username}.",
                "visibility": "PUBLIC"
            }
            area_id = create_area(session, area_req)

            survey_req = {
                "title": f"Opinions about {username} as a person (area {i})",
                "description": f"Share your honest opinion about {username}'s personality, habits and work style.",
                "areaId": area_id
            }
            survey_id = create_survey(session, survey_req)
            surveys.append((survey_id, username))

    # 3) For each survey: create exactly 10 survey responses
    user_ids_only = [u[0] for u in users]

    for survey_id, subject_name in surveys:
        num_responses = 10
        print(f"Creating {num_responses} responses for survey {survey_id} about {subject_name}")
        for _ in range(num_responses):
            author_id = random.choice(user_ids_only)
            mark = random.randint(1, 5)
            comment = generate_comment(mark, subject_name)
            response_req = {
                "surveyId": survey_id,
                "userId": author_id,
                "mark": mark,
                "comment": comment
            }
            create_survey_response(session, response_req)

if __name__ == "__main__":
    main()
