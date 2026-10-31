import csv
import random

OUTPUT_FILE = "data/raw/large_jobs.csv"

companies = [
    "ABC Technologies",
    "XYZ Solutions",
    "TechNova",
    "DataWorks",
    "Cloud Systems",
    "Innovate Labs",
    "NextGen Technologies",
    "Smart Solutions",
    "Global Tech",
    "Digital Systems"
]

roles = [
    "Python Developer",
    "Data Scientist",
    "Java Developer",
    "Machine Learning Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Software Engineer",
    "Data Analyst",
    "AI Engineer",
    "Full Stack Developer"
]

locations = [
    "Hyderabad",
    "Bangalore",
    "Chennai",
    "Pune",
    "Mumbai",
    "Delhi",
    "Visakhapatnam",
    "Kolkata",
    "Noida",
    "Gurgaon"
]

skills = [
    ["Python", "Django", "PostgreSQL"],
    ["Python", "Machine Learning", "Pandas"],
    ["Java", "Spring Boot", "MySQL"],
    ["Python", "TensorFlow", "Scikit-learn"],
    ["Python", "FastAPI", "PostgreSQL"],
    ["JavaScript", "React", "Node.js"],
    ["Python", "Flask", "MongoDB"],
    ["SQL", "Python", "Power BI"],
    ["Python", "PyTorch", "NumPy"],
    ["Java", "Spring", "Oracle"]
]

degrees = [
    "B.Tech",
    "M.Tech",
    "B.E",
    "MCA",
    "B.Sc"
]

work_modes = [
    "remote",
    "hybrid",
    "on-site"
]

rows = []

NUMBER_OF_RECORDS = 5000

for i in range(1, NUMBER_OF_RECORDS + 1):

    company = random.choice(companies)
    role = random.choice(roles)
    location = random.choice(locations)
    skill_list = random.choice(skills)
    degree = random.choice(degrees)
    work_mode = random.choice(work_modes)

    experience = random.choice([
        "1-2 years",
        "2-3 years",
        "2+ years",
        "3-5 years",
        "4+ years"
    ])

    salary = random.choice([
        "4-6 LPA",
        "5-8 LPA",
        "6-10 LPA",
        "8-12 LPA",
        "10-15 LPA"
    ])

    description = (
        f"{company} is hiring a {role} in {location}. "
        f"Candidates should have a {degree} degree and "
        f"{experience} of experience in "
        f"{', '.join(skill_list)}. "
        f"The position is {work_mode} with a salary of {salary}."
    )

    rows.append({
        "id": i,
        "job_description": description
    })


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["id", "job_description"]
    )

    writer.writeheader()
    writer.writerows(rows)


print("Large dataset created successfully.")
print("Total records:", NUMBER_OF_RECORDS)
print("File:", OUTPUT_FILE)