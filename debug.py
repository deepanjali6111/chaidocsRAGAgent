import requests
import os
from dotenv import load_dotenv

def debug_rag_issues():
    """Debug script to identify RAG issues"""
    
    print("🔍 Debugging ChaiDocs RAG Issues")
    print("=" * 50)
    
    # 1. Check environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    print("1. Environment Variables:")
    print(f"   GEMINI_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
    if api_key:
        print(f"   Key length: {len(api_key)} characters")
        print(f"   Starts with: {api_key[:10]}...")
    
    # 2. Check URL accessibility
    print("\n2. URL Accessibility:")
    urls = [
        "https://docs.chaicode.com/youtube/getting-started/",
        "https://docs.chaicode.com/youtube/authentication/",
        "https://docs.chaicode.com/youtube/api-reference/",
        "https://docs.chaicode.com/",  # Try base URL
        "https://chaicode.com/"  # Try main site
        "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
        "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
        "http://docs.chaicode.com/youtube/chai-aur-git/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
        "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
        "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
        "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
        "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
        "https://docs.chaicode.com/youtube/chai-aur-git/github/",
        "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
        "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
        "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
        "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
        "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
        "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
        "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
        "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
        "https://docs.chaicode.com/youtube/chai-aur-c/functions/"
         "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
         "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
         "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
         "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
         "https://docs.chaicode.com/youtube/chai-aur-django/models/",
         "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
         "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
         "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
         "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
         "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
         "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
         "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
         "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
         "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"   {url}: {'✅' if response.status_code == 200 else '❌'} ({response.status_code})")
        except Exception as e:
            print(f"   {url}: ❌ Error - {str(e)}")
    
    # 3. Check if Chroma DB exists
    print("\n3. Local Database:")
    chroma_path = "./chroma_db"
    if os.path.exists(chroma_path):
        print(f"   Chroma DB: ✅ Exists at {chroma_path}")
        files = os.listdir(chroma_path)
        print(f"   Files: {files}")
    else:
        print(f"   Chroma DB: ❌ Not found at {chroma_path}")
    
    # 4. Test Google AI connection
    print("\n4. Google AI Connection:")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Test with a simple generation
            model = genai.GenerativeModel('gemini-2.0-flash')
            test_response = model.generate_content("Say 'Connection successful'")
            print(f"   Connection: ✅ {test_response.text}")
        except Exception as e:
            print(f"   Connection: ❌ {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎯 Recommendations:")
    print("1. If URLs are not accessible, use fallback content or find correct URLs")
    print("2. Ensure GEMINI_API_KEY is valid and has proper permissions")
    print("3. Try deleting ./chroma_db folder to rebuild the index")
    print("4. Lower the similarity threshold or remove it entirely")
    print("5. Add more logging to see what documents are being retrieved")

if __name__ == "__main__":
    debug_rag_issues()