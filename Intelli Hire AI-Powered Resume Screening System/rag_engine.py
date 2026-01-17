# ============================================================================
# FIXED RAG ENGINE - Professional Version with Fresher Detection
# ============================================================================

import google.generativeai as genai
from typing import List, Dict, Any
import json
import re
from datetime import datetime

class RAGEngine:
    """
    RAG Engine with Gemini integration and Fresher Detection
    """
    
    def __init__(self, gemini_api_key: str):
        try:
            genai.configure(api_key=gemini_api_key)
            
            self.llm = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            
            # Test API
            print("🧪 Testing Gemini API...")
            test_response = self.llm.generate_content("Hi, respond with 'OK'")
            if test_response and test_response.text:
                print(f"✅ Gemini API Working! Response: {test_response.text}")
                self.api_working = True
            else:
                print("⚠️  API responded but no text")
                self.api_working = False
            
        except Exception as e:
            print(f"⚠️  Gemini API Error: {e}")
            print("⚠️  Will use algorithmic analysis only")
            self.api_working = False
            self.llm = None
        
        from daa_algorithms import DAAAlgorithms
        self.daa = DAAAlgorithms()
        
        print(f"✅ RAG Engine initialized (LLM: {'Active' if self.api_working else 'Disabled'})")
    
    def extract_candidate_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured candidate information with fresher detection"""
        info = {
            'name': 'Unknown',
            'email': None,
            'phone': None,
            'education': [],
            'experience_years': 0,
            'graduation_year': None,
            'is_fresher': False,
            'candidate_type': 'Unknown'
        }
        
        # Extract Name
        first_lines = resume_text.split('\n')[:5]
        for line in first_lines:
            name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', line)
            if name_match and len(name_match.group(1)) > 5:
                info['name'] = name_match.group(1).strip()
                break
        
        # Extract Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
        if email_match:
            info['email'] = email_match.group(0)
        
        # Extract Phone
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\b\d{10}\b',
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, resume_text)
            if phone_match:
                info['phone'] = phone_match.group(0)
                break
        
        # Extract Education and Graduation Year
        education_section = re.search(
            r'(?:EDUCATION|Education|ACADEMIC|QUALIFICATION)(.*?)(?:EXPERIENCE|SKILLS|PROJECTS|INTERNSHIP|$)',
            resume_text,
            re.IGNORECASE | re.DOTALL
        )
        
        graduation_years = []
        
        if education_section:
            edu_text = education_section.group(1)
            
            # Extract degrees
            degrees = re.findall(
                r'\b(B\.?Tech|B\.?E\.?|M\.?Tech|M\.?E\.?|MBA|MS|PhD|Bachelor|Master|BSc|MSc|BCA|MCA|B\.Sc|M\.Sc)[^\n]{0,100}',
                edu_text,
                re.IGNORECASE
            )
            info['education'] = [deg.strip() for deg in degrees[:3]]
            
            # Extract graduation years - MULTIPLE PATTERNS
            year_patterns = [
                r'(?:Graduation|Graduated|Passing|Pass\s*out|Completed)[\s:]*(\d{4})',  # "Graduation: 2024"
                r'(\d{4})\s*[-–—]\s*(\d{4})',  # "2020-2024" (take second year)
                r'(?:Class\s*of|Batch\s*of)\s*(\d{4})',  # "Class of 2024"
                r'\b(\d{4})\s*(?:\(|\[)?\s*(?:Expected|Pursuing|Current)?\s*(?:\)|\])?',  # "2024 (Expected)"
                r'(?:Year|Yr)[\s:]*(\d{4})',  # "Year: 2024"
            ]
            
            for pattern in year_patterns:
                matches = re.findall(pattern, edu_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # For range patterns like "2020-2024", take the end year
                        year = int(match[-1]) if match[-1] else int(match[0])
                    else:
                        year = int(match)
                    
                    # Only accept years between 2000 and 2030
                    if 2000 <= year <= 2030:
                        graduation_years.append(year)
        
        # Determine graduation year (most recent)
        if graduation_years:
            info['graduation_year'] = max(graduation_years)
        
        # Extract Experience Years - ONLY from explicit mentions
        exp_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',  # "5 years experience"
        ]
        
        years_found = []
        for pattern in exp_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    # Only accept reasonable year values (0-50)
                    if 0 <= years <= 50:
                        years_found.append(years)
                except:
                    pass
        
        if years_found:
            info['experience_years'] = max(years_found)
        
        # DETERMINE CANDIDATE TYPE (Fresher or Experienced)
        current_year = datetime.now().year
        
        if info['graduation_year']:
            years_since_graduation = current_year - info['graduation_year']
            
            # Fresher: Graduated within last 2 years AND no experience mentioned
            if years_since_graduation <= 2 and info['experience_years'] == 0:
                info['is_fresher'] = True
                info['candidate_type'] = 'Fresher'
            # Recent Graduate with some experience
            elif years_since_graduation <= 2 and info['experience_years'] > 0:
                info['is_fresher'] = False
                info['candidate_type'] = f'Recent Graduate ({info["experience_years"]} yr exp)'
            # Experienced Professional
            elif info['experience_years'] > 0:
                info['is_fresher'] = False
                info['candidate_type'] = f'Experienced ({info["experience_years"]} years)'
            # Older graduate, no mentioned experience
            elif years_since_graduation > 2:
                info['is_fresher'] = False
                info['candidate_type'] = f'Graduate ({info["graduation_year"]})'
            else:
                info['candidate_type'] = 'Unknown'
        else:
            # No graduation year found
            if info['experience_years'] > 0:
                info['is_fresher'] = False
                info['candidate_type'] = f'Experienced ({info["experience_years"]} years)'
            else:
                info['candidate_type'] = 'Unknown'
        
        return info
    
    def extract_skills_enhanced(self, text: str) -> List[str]:
        """Enhanced skill extraction"""
        text_lower = text.lower()
        found_skills = set()
        
        skill_patterns = {
            'python': r'\bpython\b',
            'java': r'\bjava\b(?!script)',
            'javascript': r'\b(javascript|js)\b',
            'typescript': r'\btypescript\b',
            'c++': r'\bc\+\+\b',
            'c#': r'\bc#\b',
            'go': r'\b(golang|go)\b',
            'rust': r'\brust\b',
            'ruby': r'\bruby\b',
            'php': r'\bphp\b',
            'html': r'\bhtml\b',
            'css': r'\bcss\b',
            'react': r'\breact\b',
            'angular': r'\bangular\b',
            'vue': r'\bvue\b',
            'node': r'\bnode\b',
            'express': r'\bexpress\b',
            'django': r'\bdjango\b',
            'flask': r'\bflask\b',
            'spring': r'\bspring\b',
            'sql': r'\bsql\b',
            'mysql': r'\bmysql\b',
            'postgresql': r'\bpostgresql\b',
            'mongodb': r'\bmongodb\b',
            'redis': r'\bredis\b',
            'aws': r'\baws\b',
            'azure': r'\bazure\b',
            'gcp': r'\bgcp\b',
            'docker': r'\bdocker\b',
            'kubernetes': r'\bkubernetes\b',
            'jenkins': r'\bjenkins\b',
            'machine learning': r'\b(machine learning|ml)\b',
            'deep learning': r'\b(deep learning|dl)\b',
            'nlp': r'\bnlp\b',
            'tensorflow': r'\btensorflow\b',
            'pytorch': r'\bpytorch\b',
            'git': r'\bgit\b',
            'github': r'\bgithub\b',
            'jira': r'\bjira\b',
            'agile': r'\bagile\b',
            'rest api': r'\brest\b',
            'graphql': r'\bgraphql\b',
            'microservices': r'\bmicroservices\b',
            'numpy': r'\bnumpy\b',
            'pandas': r'\bpandas\b',
            'data structures': r'\bdata structures\b',
            'problem solving': r'\b(problem solving|problem-solving)\b',
            'communication skills': r'\b(communication|communication skills)\b',
        }
        
        for skill_name, pattern in skill_patterns.items():
            if re.search(pattern, text_lower):
                found_skills.add(skill_name)
        
        return list(found_skills)
    
    def fuzzy_match_skills(self, resume_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
        """Fuzzy skill matching using advanced algorithms"""
        matched_skills = []
        missing_skills = []
        match_details = []
        
        resume_skills_lower = [s.lower().strip() for s in resume_skills]
        
        for req_skill in required_skills:
            req_skill_clean = req_skill.lower().strip()
            best_match = None
            best_similarity = 0.0
            match_method = ""
            algorithm_used = ""
            
            # Exact match
            if req_skill_clean in resume_skills_lower:
                best_match = req_skill_clean
                best_similarity = 100.0
                match_method = "Exact Match"
                algorithm_used = "Pattern Matching"
            
            # Substring match
            if not best_match:
                for res_skill in resume_skills_lower:
                    if req_skill_clean in res_skill or res_skill in req_skill_clean:
                        best_match = res_skill
                        best_similarity = 95.0
                        match_method = "Substring Match"
                        algorithm_used = "String Matching"
                        break
            
            # Fuzzy match using similarity algorithms
            if not best_match:
                for res_skill in resume_skills_lower:
                    similarity = self.daa.similarity_score(res_skill, req_skill_clean)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = res_skill
                        match_method = f"Fuzzy Match"
                        algorithm_used = "Similarity Analysis"
            
            if best_similarity >= 60.0:
                matched_skills.append(req_skill)
                match_details.append({
                    'required': req_skill,
                    'found': best_match,
                    'similarity': round(best_similarity, 1),
                    'method': match_method,
                    'algorithm': algorithm_used
                })
            else:
                missing_skills.append(req_skill)
                match_details.append({
                    'required': req_skill,
                    'found': 'Not Found',
                    'similarity': 0.0,
                    'method': 'No Match',
                    'algorithm': 'N/A'
                })
        
        skill_match_percentage = (len(matched_skills) / len(required_skills) * 100) if required_skills else 0
        
        return {
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'match_details': match_details,
            'skill_match_percentage': round(skill_match_percentage, 1),
            'total_required': len(required_skills),
            'total_matched': len(matched_skills),
            'total_missing': len(missing_skills)
        }
    
    def analyze_with_rag(self, resume_text: str, job_description: str, 
                         required_skills: List[str]) -> Dict[str, Any]:
        """Comprehensive RAG analysis with fresher detection"""
        print(f"\n{'='*70}")
        print(f"🔍 STARTING ANALYSIS")
        print(f"{'='*70}")
        
        # Step 1: Extract info
        print("📋 Extracting candidate information...")
        candidate_info = self.extract_candidate_info(resume_text)
        print(f"   ✅ Name: {candidate_info['name']}")
        print(f"   ✅ Type: {candidate_info['candidate_type']}")
        if candidate_info['graduation_year']:
            print(f"   ✅ Graduation: {candidate_info['graduation_year']}")
        
        # Step 2: Extract skills
        print("\n🔎 Extracting skills...")
        resume_skills = self.extract_skills_enhanced(resume_text)
        print(f"   ✅ Found {len(resume_skills)} skills")
        
        # Step 3: Match skills
        print("\n🧮 Running matching algorithms...")
        match_result = self.fuzzy_match_skills(resume_skills, required_skills)
        print(f"   ✅ Matched: {match_result['total_matched']}/{match_result['total_required']} ({match_result['skill_match_percentage']}%)")
        
        # Step 4: LLM Analysis
        llm_analysis = None
        if self.api_working and self.llm:
            print("\n🤖 Running AI analysis...")
            try:
                llm_analysis = self._try_llm_analysis(
                    resume_text, job_description, candidate_info, match_result, required_skills
                )
                if llm_analysis:
                    print("   ✅ AI analysis completed")
            except Exception as e:
                print(f"   ⚠️  AI error: {e}")
        
        # Build final result
        if llm_analysis:
            final = self._merge_analysis(candidate_info, match_result, llm_analysis, resume_skills)
        else:
            print("\n💡 Using intelligent algorithmic scoring...")
            final = self._smart_algorithmic_scoring(candidate_info, match_result, resume_skills)
        
        print(f"\n{'='*70}")
        print(f"✅ COMPLETED - Overall: {final['overall_score']}%")
        print(f"{'='*70}\n")
        
        return final
    
    def _try_llm_analysis(self, resume_text: str, job_description: str,
                         candidate_info: Dict, match_result: Dict,
                         required_skills: List[str]) -> Dict:
        """Try LLM analysis"""
        try:
            prompt = f"""Analyze this candidate:

JOB: {job_description[:400]}
REQUIRED: {', '.join(required_skills)}

CANDIDATE:
Name: {candidate_info['name']}
Type: {candidate_info['candidate_type']}
Graduation Year: {candidate_info.get('graduation_year', 'Not specified')}
Education: {', '.join(candidate_info['education'][:2])}
Skills Matched: {match_result['total_matched']}/{match_result['total_required']}

RESUME: {resume_text[:800]}...

JSON response:
{{
    "overall_score": 0-100,
    "experience_match_score": 0-100,
    "qualification_score": 0-100,
    "cultural_fit_score": 0-100,
    "strengths": ["2-3 items"],
    "weaknesses": ["1-2 items"],
    "ai_summary": "Brief summary",
    "recommendation": "HIGHLY_RECOMMENDED/RECOMMENDED/MAYBE/NOT_RECOMMENDED",
    "reasoning": "Why",
    "suggested_questions": ["2 questions"]
}}"""
            
            response = self.llm.generate_content(prompt)
            
            if response and response.text:
                return self._parse_llm_response(response.text)
            
        except Exception as e:
            print(f"   LLM error: {e}")
        
        return None
    
    def _parse_llm_response(self, text: str) -> Dict:
        """Parse JSON from LLM"""
        try:
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
                json_str = json_match.group(0) if json_match else text
            
            return json.loads(json_str)
        except:
            return None
    
    def _merge_analysis(self, candidate_info: Dict, match_result: Dict,
                       llm_analysis: Dict, resume_skills: List[str]) -> Dict:
        """Merge LLM with algorithmic results"""
        return {
            'candidate_name': candidate_info['name'],
            'email': candidate_info['email'],
            'phone': candidate_info['phone'],
            'education': candidate_info['education'],
            'graduation_year': candidate_info['graduation_year'],
            'is_fresher': candidate_info['is_fresher'],
            'candidate_type': candidate_info['candidate_type'],
            'experience_years': candidate_info['experience_years'],
            'matched_skills': match_result['matched_skills'],
            'missing_skills': match_result['missing_skills'],
            'skill_match_percentage': match_result['skill_match_percentage'],
            'match_details': match_result['match_details'],
            'total_skills_found': len(resume_skills),
            'overall_score': llm_analysis.get('overall_score', 50),
            'experience_match_score': llm_analysis.get('experience_match_score', 50),
            'qualification_score': llm_analysis.get('qualification_score', 50),
            'cultural_fit_score': llm_analysis.get('cultural_fit_score', 50),
            'strengths': llm_analysis.get('strengths', []),
            'weaknesses': llm_analysis.get('weaknesses', []),
            'ai_summary': llm_analysis.get('ai_summary', ''),
            'detailed_analysis': llm_analysis.get('reasoning', ''),
            'recommendation': llm_analysis.get('recommendation', 'MAYBE'),
            'reasoning': llm_analysis.get('reasoning', ''),
            'suggested_questions': llm_analysis.get('suggested_questions', []),
            'analysis_timestamp': datetime.now().isoformat(),
            'llm_used': True
        }
    
    def _smart_algorithmic_scoring(self, candidate_info: Dict, match_result: Dict,
                                   resume_skills: List[str]) -> Dict:
        """Smart scoring without LLM"""
        
        skill_score = match_result['skill_match_percentage']
        exp_years = candidate_info['experience_years']
        has_education = len(candidate_info['education']) > 0
        is_fresher = candidate_info['is_fresher']
        
        # Experience scoring
        if is_fresher:
            # For freshers, don't penalize lack of experience
            experience_score = 60.0  # Base score for freshers
        elif exp_years == 0:
            experience_score = 20.0
        else:
            experience_score = min(20 + (exp_years * 16), 100)
        
        # Education scoring
        if has_education:
            edu_text = str(candidate_info['education']).lower()
            if any(k in edu_text for k in ['master', 'phd', 'mba', 'm.tech', 'ms']):
                education_score = 90.0
            else:
                education_score = 70.0
        else:
            education_score = 40.0
        
        # Overall: 40% skills + 40% experience + 20% education
        overall_score = (skill_score * 0.4) + (experience_score * 0.4) + (education_score * 0.2)
        overall_score = round(min(overall_score, 100), 1)
        
        # Generate strengths/weaknesses
        strengths = []
        weaknesses = []
        
        if skill_score >= 70:
            strengths.append(f"Strong skill match: {match_result['total_matched']}/{match_result['total_required']} required skills")
        elif skill_score >= 50:
            strengths.append(f"Good skill foundation: {match_result['total_matched']} core skills")
        else:
            weaknesses.append(f"Limited skill match: Only {match_result['total_matched']}/{match_result['total_required']} required skills")
        
        # Add candidate type to strengths
        if is_fresher:
            strengths.append(f"Fresh talent with recent education ({candidate_info['graduation_year']})")
        elif exp_years >= 3:
            strengths.append(f"Solid {exp_years} years of professional experience")
        elif exp_years > 0:
            strengths.append(f"{exp_years} years of relevant experience")
        else:
            if candidate_info['graduation_year']:
                current_year = datetime.now().year
                years_since = current_year - candidate_info['graduation_year']
                if years_since > 2:
                    weaknesses.append(f"Graduated in {candidate_info['graduation_year']} with no mentioned experience")
        
        if has_education:
            strengths.append(f"Qualified: {candidate_info['education'][0]}")
        
        if match_result['total_missing'] > 0:
            missing_list = ', '.join(match_result['missing_skills'][:3])
            weaknesses.append(f"Missing {match_result['total_missing']} skills: {missing_list}")
        
        # Recommendation
        if overall_score >= 75:
            recommendation = "HIGHLY_RECOMMENDED"
            reasoning = f"Excellent candidate with {overall_score}% match - strong skills and qualifications"
        elif overall_score >= 60:
            recommendation = "RECOMMENDED"
            reasoning = f"Good candidate with {overall_score}% match - worth detailed interview"
        elif overall_score >= 40:
            recommendation = "MAYBE"
            reasoning = f"Moderate fit at {overall_score}% - consider based on other factors"
        else:
            recommendation = "NOT_RECOMMENDED"
            reasoning = f"Below requirements at {overall_score}% match"
        
        return {
            'candidate_name': candidate_info['name'],
            'email': candidate_info['email'],
            'phone': candidate_info['phone'],
            'education': candidate_info['education'],
            'graduation_year': candidate_info['graduation_year'],
            'is_fresher': is_fresher,
            'candidate_type': candidate_info['candidate_type'],
            'experience_years': exp_years,
            'matched_skills': match_result['matched_skills'],
            'missing_skills': match_result['missing_skills'],
            'skill_match_percentage': skill_score,
            'match_details': match_result['match_details'],
            'total_skills_found': len(resume_skills),
            'overall_score': overall_score,
            'experience_match_score': experience_score,
            'qualification_score': education_score,
            'cultural_fit_score': 50.0,
            'strengths': strengths if strengths else ["Analysis completed"],
            'weaknesses': weaknesses if weaknesses else ["Manual review recommended"],
            'ai_summary': f"{candidate_info['name']} - {candidate_info['candidate_type']}. {skill_score:.1f}% skill match. {recommendation.replace('_', ' ')}.",
            'detailed_analysis': f"Comprehensive analysis: Skills={skill_score:.1f}%, Experience={experience_score:.1f}%, Education={education_score:.1f}%. {reasoning}",
            'recommendation': recommendation,
            'reasoning': reasoning,
            'suggested_questions': [
                f"Tell us about your experience with {match_result['matched_skills'][0]}" if match_result['matched_skills'] else "Describe your technical experience",
                f"How would you approach learning {match_result['missing_skills'][0]}?" if match_result['missing_skills'] else "What technologies interest you?",
                "Walk us through a recent challenging project" if not is_fresher else "Tell us about your academic projects"
            ],
            'analysis_timestamp': datetime.now().isoformat(),
            'llm_used': False
        }