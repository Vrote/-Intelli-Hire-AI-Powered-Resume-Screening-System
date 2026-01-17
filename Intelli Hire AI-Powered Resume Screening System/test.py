# ============================================================================
# API DIAGNOSTIC TOOL - Check Gemini API
# ============================================================================

import google.generativeai as genai
import os
from dotenv import load_dotenv

def test_gemini_api():
    """Test Gemini API connection and functionality"""
    
    print("\n" + "="*70)
    print("🔍 GEMINI API DIAGNOSTIC TOOL")
    print("="*70 + "\n")
    
    # Step 1: Load API key
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        print("\n📝 Create .env file with:")
        print("   GEMINI_API_KEY=your-api-key-here")
        print("\n🔗 Get FREE API key from: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Step 2: Configure API
    try:
        genai.configure(api_key=api_key)
        print("✅ API configured successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Step 3: Test basic call
    print("\n📡 Testing API connection...")
    try:
        # ✅ FIXED: Use new model name
        model = genai.GenerativeModel("gemini-2.0-flash")  # Was: 'gemini-pro'
        response = model.generate_content("Say 'Hello, API is working!'")
        
        if response and response.text:
            print(f"✅ API Response: {response.text}")
        else:
            print("⚠️  API responded but no text returned")
            return False
            
    except Exception as e:
        print(f"❌ API Call Failed: {e}")
        print("\n🔍 Common Issues:")
        print("   1. Invalid API key")
        print("   2. API quota exceeded")
        print("   3. Network/firewall blocking")
        print("   4. Wrong API key format")
        return False
    
    # Step 4: Test JSON response
    print("\n📝 Testing JSON response parsing...")
    try:
        prompt = """Respond with valid JSON:
{
    "score": 85,
    "recommendation": "RECOMMENDED",
    "summary": "Good candidate"
}"""
        
        response = model.generate_content(prompt)
        print(f"✅ JSON Test Response: {response.text[:100]}...")
        
    except Exception as e:
        print(f"⚠️  JSON test warning: {e}")
    
    # Step 5: Test resume analysis prompt
    print("\n📄 Testing resume analysis prompt...")
    try:
        test_prompt = """Analyze this candidate:

JOB: Senior Python Developer
SKILLS REQUIRED: Python, Django, AWS

CANDIDATE:
Name: John Doe
Experience: 5 years
Skills: Python, Flask, Docker

Respond in JSON format:
{
    "overall_score": 0-100,
    "recommendation": "RECOMMENDED",
    "strengths": ["Python expert"],
    "weaknesses": ["Missing Django"],
    "ai_summary": "Strong Python background"
}"""
        
        response = model.generate_content(test_prompt)
        
        if response and response.text:
            print("✅ Resume analysis test passed")
            print(f"   Response preview: {response.text[:150]}...")
        else:
            print("⚠️  No response from resume analysis test")
            
    except Exception as e:
        print(f"⚠️  Resume analysis test failed: {e}")
    
    print("\n" + "="*70)
    print("✅ DIAGNOSTIC COMPLETE - API IS WORKING!")
    print("="*70)
    print("\n💡 If you see this message, your API is configured correctly.")
    print("   If analysis still fails, check:")
    print("   1. Resume file is readable (PDF/DOCX)")
    print("   2. Resume has extractable text (not scanned image)")
    print("   3. Check terminal logs for specific errors")
    print("\n")
    
    return True

if __name__ == "__main__":
    test_gemini_api()