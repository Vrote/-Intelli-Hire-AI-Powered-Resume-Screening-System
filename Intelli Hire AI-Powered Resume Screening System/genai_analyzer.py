import google.generativeai as genai
import json
import re
from typing import Dict, List, Any

class GenAIAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize Gemini AI
        Get free API key from: https://makersuite.google.com/app/apikey
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def analyze_resume(self, resume_text: str, job_description: str, 
                      required_skills: List[str]) -> Dict[str, Any]:
        """
        Use Gemini AI to deeply analyze resume against job requirements
        """
        
        prompt = f"""
You are an expert HR recruiter with 15+ years of experience. Analyze this resume against the job requirements and provide a comprehensive evaluation.

JOB REQUIREMENTS:
Title: Job Position
Description: {job_description}
Required Skills: {', '.join(required_skills)}

CANDIDATE RESUME:
{resume_text}

Provide your analysis in the following JSON format (must be valid JSON):
{{
    "candidate_name": "extracted name or 'Unknown'",
    "email": "extracted email or null",
    "phone": "extracted phone or null",
    "experience_years": estimated years as number,
    "education": {{
        "highest_degree": "degree name",
        "institution": "university name",
        "field": "field of study"
    }},
    
    "overall_score": score from 0-100,
    "skill_match_score": score from 0-100,
    "experience_match_score": score from 0-100,
    "cultural_fit_score": score from 0-100,
    
    "skills_found": ["skill1", "skill2", ...],
    "matched_skills": ["skills that match requirements"],
    "missing_skills": ["required skills not found"],
    
    "strengths": [
        "Key strength 1",
        "Key strength 2",
        "Key strength 3"
    ],
    
    "weaknesses": [
        "Area for improvement 1",
        "Area for improvement 2"
    ],
    
    "ai_summary": "2-3 sentence professional summary of candidate",
    
    "recommendation": "HIGHLY_RECOMMENDED" or "RECOMMENDED" or "MAYBE" or "NOT_RECOMMENDED",
    
    "reasoning": "detailed explanation of scores and recommendation",
    
    "suggested_questions": [
        "Interview question 1 based on resume gaps",
        "Interview question 2 to verify claims",
        "Interview question 3 for skill assessment"
    ]
}}

Be thorough, honest, and professional. Consider:
- Actual work experience relevance, not just keywords
- Project complexity and impact
- Leadership and teamwork indicators
- Communication quality in resume
- Career progression and growth
- Red flags (gaps, job hopping, inconsistencies)
"""
        
        try:
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = result_text
            
            # Parse JSON
            analysis = json.loads(json_str)
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response: {result_text[:500]}")
            return self._get_fallback_analysis()
        except Exception as e:
            print(f"GenAI Error: {e}")
            return self._get_fallback_analysis()
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback response if AI fails"""
        return {
            "candidate_name": "Unknown",
            "email": None,
            "phone": None,
            "experience_years": 0,
            "education": {},
            "overall_score": 50.0,
            "skill_match_score": 50.0,
            "experience_match_score": 50.0,
            "cultural_fit_score": 50.0,
            "skills_found": [],
            "matched_skills": [],
            "missing_skills": [],
            "strengths": ["Analysis failed - manual review needed"],
            "weaknesses": [],
            "ai_summary": "AI analysis unavailable. Please review manually.",
            "recommendation": "MAYBE",
            "reasoning": "Automated analysis failed",
            "suggested_questions": []
        }
    
    def compare_candidates(self, candidates: List[Dict]) -> str:
        """
        AI-powered comparison of multiple candidates
        """
        prompt = f"""
You are an expert recruiter. Compare these candidates and provide insights:

CANDIDATES:
{json.dumps(candidates, indent=2)}

Provide:
1. Ranking with reasoning (who is best and why)
2. Key differentiators between top candidates
3. Hiring recommendation
4. Any red flags across candidates

Keep response concise and actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Comparison failed: {str(e)}"