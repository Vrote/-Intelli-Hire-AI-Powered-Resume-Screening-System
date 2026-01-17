# ============================================================================
# UPDATED app.py - With Fresher Detection Feature
# ============================================================================

from flask import Flask, render_template, request, jsonify
from resume_processor import ResumeProcessor
from rag_engine import RAGEngine
from supabase import create_client
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    print("⚠️  WARNING: Missing environment variables!")
    print("   Create a .env file with:")
    print("   SUPABASE_URL=your-url")
    print("   SUPABASE_KEY=your-key")
    print("   GEMINI_API_KEY=your-key")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
processor = ResumeProcessor()
rag_engine = RAGEngine(GEMINI_API_KEY)

print("\n" + "="*70)
print("✅ INTELLI-HIRE RAG SYSTEM STARTED")
print("="*70)
print(f"🌐 Server running on: http://localhost:5000")
print(f"📊 RAG Engine: Active")
print(f"🎓 Fresher Detection: Enabled")
print(f"🧮 Advanced Algorithms: Pattern Matching + Similarity Analysis")
print("="*70 + "\n")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get all jobs"""
    try:
        response = supabase.table('jobs').select('*').order('created_at', desc=True).execute()
        return jsonify({'success': True, 'jobs': response.data})
    except Exception as e:
        print(f"❌ Error fetching jobs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Create new job posting"""
    try:
        data = request.json
        
        # Parse required skills
        required_skills = [s.strip() for s in data['required_skills'].split(',') if s.strip()]
        
        job_data = {
            'title': data['title'],
            'description': data['description'],
            'required_skills': required_skills,
            'experience_required': data.get('experience_required', '')
        }
        
        print(f"\n📝 Creating job: {data['title']}")
        print(f"   Required skills: {', '.join(required_skills)}")
        
        response = supabase.table('jobs').insert(job_data).execute()
        
        print(f"   ✅ Job created with ID: {response.data[0]['id']}")
        
        return jsonify({'success': True, 'job': response.data[0]})
        
    except Exception as e:
        print(f"❌ Error creating job: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_resumes():
    """Analyze resumes using RAG with fresher detection"""
    try:
        job_id = request.form.get('job_id')
        files = request.files.getlist('resumes')
        
        if not job_id or not files:
            return jsonify({'success': False, 'error': 'Missing job_id or resume files'}), 400
        
        # Fetch job details
        job = supabase.table('jobs').select('*').eq('id', job_id).execute()
        if not job.data:
            return jsonify({'success': False, 'error': 'Job not found'}), 404
        
        job_data = job.data[0]
        
        print(f"\n{'='*70}")
        print(f"🚀 ANALYZING RESUMES FOR: {job_data['title']}")
        print(f"{'='*70}")
        print(f"📋 Required Skills: {', '.join(job_data['required_skills'])}")
        print(f"📄 Resumes to analyze: {len(files)}")
        print(f"{'='*70}")
        
        results = []
        
        for idx, file in enumerate(files, 1):
            if not file.filename:
                continue
            
            print(f"\n[{idx}/{len(files)}] Processing: {file.filename}")
            print("-" * 70)
            
            # Extract text from resume
            resume_text = processor.extract_text(file)
            
            if not resume_text or len(resume_text) < 100:
                print(f"   ⚠️  Could not extract text or text too short")
                continue
            
            print(f"   ✅ Text extracted: {len(resume_text)} characters")
            
            # Run comprehensive RAG analysis
            analysis = rag_engine.analyze_with_rag(
                resume_text,
                job_data['description'],
                job_data['required_skills']
            )
            
            # Prepare candidate data for database
            candidate_data = {
                'job_id': job_id,
                'filename': file.filename,
                
                # Candidate Info
                'candidate_name': analysis.get('candidate_name', 'Unknown'),
                'email': analysis.get('email'),
                'phone': analysis.get('phone'),
                'experience_years': analysis.get('experience_years', 0),
                'education': analysis.get('education', []),
                
                # NEW: Fresher Detection Fields
                'graduation_year': analysis.get('graduation_year'),
                'is_fresher': analysis.get('is_fresher', False),
                'candidate_type': analysis.get('candidate_type', 'Unknown'),
                
                # Scores
                'overall_score': analysis['overall_score'],
                'skill_match_score': analysis['skill_match_percentage'],
                'experience_match_score': analysis.get('experience_match_score', 50),
                'cultural_fit_score': analysis.get('cultural_fit_score', 50),
                
                # Skills
                'skills_found': analysis.get('matched_skills', []) + analysis.get('missing_skills', []),
                'matched_skills': analysis.get('matched_skills', []),
                'missing_skills': analysis.get('missing_skills', []),
                
                # Analysis
                'ai_summary': analysis.get('ai_summary', ''),
                'strengths': analysis.get('strengths', []),
                'weaknesses': analysis.get('weaknesses', []),
                'recommendation': analysis.get('recommendation', 'MAYBE'),
                'suggested_questions': analysis.get('suggested_questions', [])
            }
            
            # Save to database
            db_response = supabase.table('candidates').insert(candidate_data).execute()
            
            # Add complete analysis for response
            results.append({
                **candidate_data,
                'match_details': analysis.get('match_details', []),
                'detailed_analysis': analysis.get('detailed_analysis', ''),
                'reasoning': analysis.get('reasoning', ''),
                'required_skills': job_data['required_skills'],
                'total_skills_found': analysis.get('total_skills_found', 0)
            })
            
            print(f"   💾 Saved to database")
            print(f"   📊 Final Score: {analysis['overall_score']}%")
            print(f"   🎓 Type: {analysis.get('candidate_type', 'Unknown')}")
        
        # Sort by overall score
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        print(f"\n{'='*70}")
        print(f"✅ ANALYSIS COMPLETE - {len(results)} candidates processed")
        print(f"{'='*70}\n")
        
        return jsonify({'success': True, 'candidates': results})
        
    except Exception as e:
        print(f"\n❌ ERROR IN ANALYSIS:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/candidates/<job_id>', methods=['GET'])
def get_candidates(job_id):
    """Get all candidates for a specific job"""
    try:
        # Fetch job details
        job = supabase.table('jobs').select('*').eq('id', job_id).execute()
        
        # Fetch candidates
        candidates = supabase.table('candidates').select('*').eq('job_id', job_id).order('overall_score', desc=True).execute()
        
        # Add required skills to each candidate
        for c in candidates.data:
            c['required_skills'] = job.data[0]['required_skills']
        
        return jsonify({'success': True, 'candidates': candidates.data})
        
    except Exception as e:
        print(f"❌ Error fetching candidates: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/candidate/<candidate_id>', methods=['GET'])
def get_candidate_details(candidate_id):
    """Get detailed information for a specific candidate"""
    try:
        candidate = supabase.table('candidates').select('*').eq('id', candidate_id).execute()
        
        if not candidate.data:
            return jsonify({'success': False, 'error': 'Candidate not found'}), 404
        
        return jsonify({'success': True, 'candidate': candidate.data[0]})
        
    except Exception as e:
        print(f"❌ Error fetching candidate: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        jobs_count = supabase.table('jobs').select('id', count='exact').execute()
        candidates_count = supabase.table('candidates').select('id', count='exact').execute()
        
        # Get fresher vs experienced breakdown
        all_candidates = supabase.table('candidates').select('is_fresher').execute()
        freshers = sum(1 for c in all_candidates.data if c.get('is_fresher', False))
        experienced = len(all_candidates.data) - freshers
        
        return jsonify({
            'success': True,
            'stats': {
                'total_jobs': len(jobs_count.data),
                'total_candidates': len(candidates_count.data),
                'freshers': freshers,
                'experienced': experienced
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')