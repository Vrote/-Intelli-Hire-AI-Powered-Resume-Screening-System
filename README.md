# -Intelli-Hire-AI-Powered-Resume-Screening-System

Overview:

Intelli-Hire is an intelligent recruitment automation system that analyzes, scores, and ranks resumes against dynamic job requirements using DAA algorithms, NLP techniques, and Generative AI (RAG).
Unlike traditional keyword-based ATS systems, Intelli-Hire applies semantic similarity, fuzzy matching, and fresher-aware scoring to ensure fair and accurate candidate evaluation.


Key Features:

Dynamic job creation with custom required skills.
Resume parsing for PDF and DOCX formats.
Fresher vs Experienced candidate classification.
Fuzzy skill matching using KMP + Edit Distance.
Semantic similarity using TF-IDF + Cosine Similarity.
RAG-based AI analysis using Google Gemini.
Algorithmic fallback scoring (AI-independent).
Batch resume analysis and automatic ranking.
Persistent storage with Supabase (PostgreSQL).


Algorithms Used:

1. Knuth-Morris-Pratt (KMP) Algorithm.
   Used for fast and efficient exact skill matching.
   Time Complexity: O(n + m)
2. Levenshtein Distance (Dynamic Programming)
   Used for fuzzy skill matching and typo tolerance.
   Time Complexity: O(m × n)
3. TF-IDF + Cosine Similarity
   Measures semantic similarity between resumes and job descriptions.
   Avoids false negatives caused by different wording.
   

Setup & Installation:

Prerequisites:<br>
Python, Supabase account, Google Gemini API key<br><br>
Environment Variables (.env):<br>
SUPABASE_URL=your_supabase_url<br>
SUPABASE_KEY=your_supabase_anon_key<br>
GEMINI_API_KEY=your_gemini_api_key<br><br>
Install Dependencies:pip install -r requirements.txt, Run Application:, python app.py


System Architecture:

Recruiter creates a job with required skills.
Resumes uploaded (PDF/DOCX).
Text extraction and preprocessing.
Skill extraction and fuzzy matching (DAA).
Fresher detection and experience analysis.
RAG-based AI analysis (Gemini).
Algorithmic fallback scoring (if AI fails).
Candidate ranking and database storage.
