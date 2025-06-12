from linkedin_api import Linkedin
import os

def get_client(l_email, l_password):
    return Linkedin(l_email, l_password, debug=True)

def search_jobs(keywords: str, limit: int = 3, offset: int = 0, location: str = '', l_email = os.getenv("LINKEDIN_EMAIL"), l_password = os.getenv("LINKEDIN_PASSWORD")) -> list:
    """
    Search for jobs on LinkedIn.
    
    :param keywords: Job search keywords
    :param limit: Maximum number of job results
    :param location: Optional location filter
    :return: List of job details
    """
    try:
        client = get_client(l_email, l_password)
        jobs = client.search_jobs(
            keywords=keywords,
            location_name=location,
            limit=limit,
            offset=offset,
        )
        job_results = []
        for job in jobs:
            job_id = job["entityUrn"].split(":")[-1]
            job_data = client.get_job(job_id=job_id)

            job_title = job_data["title"]
            company_name = job_data["companyDetails"]["com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany"]["companyResolutionResult"]["name"]
            job_description = job_data["description"]["text"]
            job_location = job_data["formattedLocation"]

            job_results.append({
                "title": job_title,
                "description": job_description,
                "location": job_location,
                "company": company_name,
            })

        return job_results
    except Exception as e:
        raise Exception(f"Error while making linkedin query: {e}")