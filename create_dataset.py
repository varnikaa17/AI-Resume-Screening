import pandas as pd
import random

random.seed(42)

jobs = {
    "Data Scientist": {
        "description": "Data Scientist required with Python, SQL, machine learning, statistics, pandas, numpy and scikit-learn experience.",
        "skills": ["Python", "SQL", "Machine Learning", "Statistics", "Pandas", "NumPy", "Scikit-learn"]
    },
    "Frontend Developer": {
        "description": "Frontend Developer required with HTML, CSS, JavaScript, React, responsive design and Git experience.",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Responsive Design", "Git"]
    },
    "Backend Developer": {
        "description": "Backend Developer required with Python, Java, Node.js, REST API, SQL, databases and authentication experience.",
        "skills": ["Python", "Java", "Node.js", "REST API", "SQL", "Databases", "Authentication"]
    },
    "Cybersecurity Analyst": {
        "description": "Cybersecurity Analyst required with networking, Linux, penetration testing, vulnerability assessment, SIEM and Wireshark experience.",
        "skills": ["Networking", "Linux", "Penetration Testing", "Vulnerability Assessment", "SIEM", "Wireshark"]
    },
    "Machine Learning Engineer": {
        "description": "Machine Learning Engineer required with Python, machine learning, deep learning, TensorFlow, PyTorch and NLP experience.",
        "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP"]
    },
    "Java Developer": {
        "description": "Java Developer required with Java, OOP, Spring Boot, SQL, REST API and Git experience.",
        "skills": ["Java", "OOP", "Spring Boot", "SQL", "REST API", "Git"]
    }
}

education = [
    "B.Tech in Computer Science",
    "B.Tech in Information Technology",
    "BCA in Computer Applications",
    "M.Tech in Computer Science"
]

experience = [
    "1 year of experience",
    "2 years of experience",
    "3 years of experience",
    "4 years of experience",
    "internship experience"
]

def create_resume(skills):
    selected = random.sample(skills, random.randint(4, len(skills)))

    extra = random.sample([
        "Problem Solving",
        "Communication",
        "Teamwork",
        "Git",
        "Data Structures",
        "OOP"
    ], 2)

    selected += extra

    return (
        f"{random.choice(education)}. "
        f"{random.choice(experience)}. "
        f"Experienced in {', '.join(selected)}. "
        f"Worked on multiple projects involving "
        f"{', '.join(selected[:4])}."
    )

rows = []

for role, job in jobs.items():

    for _ in range(1000):

        # Suitable candidate
        resume = create_resume(job["skills"])

        rows.append([
            resume,
            job["description"],
            1
        ])

        # Non-suitable candidate
        other_roles = [r for r in jobs if r != role]
        other_role = random.choice(other_roles)

        wrong_resume = create_resume(
            jobs[other_role]["skills"]
        )

        rows.append([
            wrong_resume,
            job["description"],
            0
        ])

df = pd.DataFrame(
    rows,
    columns=[
        "resume_text",
        "job_description",
        "label"
    ]
)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

df.to_csv(
    "data/resume_job_dataset.csv",
    index=False
)

print("Dataset created successfully!")
print("Total rows:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())