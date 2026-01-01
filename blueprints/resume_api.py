"""
Resume Screener API Blueprint
ATS-powered resume analysis and job description matching
"""

from flask import Blueprint, request, jsonify
import PyPDF2
import docx
import io
import base64
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Create Blueprint
resume_bp = Blueprint('resume', __name__)

# Download required NLTK data (only once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📥 Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# Load spaCy model
print("🧠 Loading spaCy model for resume analysis...")
try:
    nlp = spacy.load('en_core_web_sm')
    print("✅ spaCy model loaded successfully!")
except OSError:
    print("⚠️  spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


class ResumeAnalyzer:
    """ATS Resume Analyzer with comprehensive skill matching"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Expanded technical skills database
        self.tech_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'typescript', 'php', 'swift', 'kotlin', 'perl', 'powershell', 'bash', 'sql'],
            'web': ['react', 'angular', 'vue', 'node.js', 'nodejs', 'express', 'django', 'flask', 'spring', 'asp.net', 'next.js', 'nextjs', 'html', 'css', 'fastapi', 'streamlit'],
            'database': ['mongodb', 'postgresql', 'mysql', 'oracle', 'redis', 'cassandra', 'dynamodb', 'firebase', 'nosql', 'sql'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd', 'vercel', 'render', 'heroku', 'netlify'],
            'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'nlp', 'opencv', 'keras', 'hugging face', 'transformers', 'lstm', 'cnn', 'neural networks', 'neural network', 'convolutional'],
            'data': ['data analysis', 'data science', 'data visualization', 'matplotlib', 'seaborn', 'tableau', 'power bi', 'excel', 'jupyter', 'colab', 'eda', 'exploratory data analysis'],
            'tools': ['git', 'github', 'gitlab', 'jira', 'agile', 'scrum', 'linux', 'unix', 'windows', 'vscode', 'intellij', 'jupyter', 'vs code'],
            
            # Cybersecurity specific skills
            'security_tools': ['wireshark', 'nmap', 'metasploit', 'burp suite', 'splunk', 'qradar', 'snort', 'nessus', 'kali linux', 'oscp', 'owasp'],
            'security_concepts': ['penetration testing', 'vulnerability assessment', 'threat detection', 'incident response', 'siem', 'firewall', 'ids', 'ips', 'vpn', 'encryption', 'cryptography', 'pki', 'ssl', 'tls'],
            'security_protocols': ['tcp/ip', 'dns', 'http', 'https', 'ssh', 'ftp', 'smtp', 'network security', 'web security'],
            'security_compliance': ['iso 27001', 'nist', 'pci-dss', 'pci dss', 'gdpr', 'hipaa', 'compliance', 'risk assessment', 'security audit'],
            'security_certifications': ['ceh', 'cissp', 'security+', 'comptia', 'cysa+', 'oscp', 'cism', 'ethical hacker'],
            'security_practices': ['malware analysis', 'forensics', 'digital forensics', 'breach response', 'security hardening', 'access control', 'authentication', 'authorization']
        }
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_keywords(self, text, top_n=20):
        """Extract important keywords using TF-IDF"""
        cleaned_text = self.clean_text(text)
        
        try:
            vectorizer = TfidfVectorizer(max_features=top_n, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
            return [kw[0] for kw in keywords]
        except:
            words = cleaned_text.split()
            words = [w for w in words if w not in self.stop_words and len(w) > 3]
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(top_n)]
    
    def calculate_similarity(self, resume_text, job_description):
        """Calculate cosine similarity between resume and JD"""
        cleaned_resume = self.clean_text(resume_text)
        cleaned_jd = self.clean_text(job_description)
        
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return round(similarity * 100, 2)
    
    def extract_skills(self, text):
        """Extract technical skills from text"""
        text_lower = text.lower()
        text_normalized = text_lower.replace('-', ' ').replace('.', ' ')
        found_skills = {}
        
        for category, skills in self.tech_skills.items():
            found = []
            for skill in skills:
                skill_normalized = skill.replace('-', ' ').replace('.', ' ')
                
                # Exact match
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(skill)
                    continue
                
                # Normalized match
                pattern_normalized = r'\b' + re.escape(skill_normalized) + r'\b'
                if re.search(pattern_normalized, text_normalized):
                    found.append(skill)
                    continue
                
                # Special cases
                if skill == 'scikit-learn' and 'sklearn' in text_lower:
                    found.append(skill)
                elif skill == 'node.js' and 'nodejs' in text_lower:
                    found.append(skill)
                elif skill == 'next.js' and 'nextjs' in text_lower:
                    found.append(skill)
                    
            if found:
                found_skills[category] = list(set(found))
                
        return found_skills
    
    def find_missing_skills(self, resume_text, job_description):
        """Find skills in JD that are missing in resume"""
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(job_description)
        
        missing = {}
        for category, skills in jd_skills.items():
            resume_category_skills = resume_skills.get(category, [])
            missing_in_category = [s for s in skills if s not in resume_category_skills]
            if missing_in_category:
                missing[category] = missing_in_category
                
        return missing
    
    def extract_experience(self, text):
        """Extract years of experience mentioned"""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return None
    
    def calculate_ats_score(self, resume_text, job_description):
        """Calculate overall ATS score"""
        scores = {}
        
        # 1. Keyword Match (40%)
        similarity = self.calculate_similarity(resume_text, job_description)
        scores['keyword_match'] = similarity
        
        # 2. Skills Match (30%)
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(job_description)
        
        total_jd_skills = sum(len(skills) for skills in jd_skills.values())
        matched_skills = 0
        
        for category, skills in jd_skills.items():
            resume_category_skills = resume_skills.get(category, [])
            matched_skills += len([s for s in skills if s in resume_category_skills])
        
        skills_score = (matched_skills / total_jd_skills * 100) if total_jd_skills > 0 else 0
        scores['skills_match'] = round(skills_score, 2)
        
        # 3. Format & Structure (30%)
        sections = ['experience', 'education', 'skills', 'projects']
        found_sections = sum(1 for section in sections if section in resume_text.lower())
        format_score = (found_sections / len(sections)) * 100
        
        scores['format_score'] = round(format_score, 2)
        
        # Overall ATS Score
        overall = (
            scores['keyword_match'] * 0.4 +
            scores['skills_match'] * 0.3 +
            scores['format_score'] * 0.3
        )
        scores['overall'] = round(overall, 2)
        
        return scores
    
    def generate_suggestions(self, resume_text, job_description, missing_skills, scores):
        """Generate improvement suggestions"""
        suggestions = []
        
        if scores['overall'] < 40:
            suggestions.append("⚠️ CRITICAL: Your resume is not a good match for this role. Consider applying to positions that better align with your skills.")
        
        if missing_skills:
            total_missing = sum(len(skills) for skills in missing_skills.values())
            
            if total_missing > 15:
                suggestions.append(f"❌ {total_missing} required skills are missing. This role requires a completely different skill set.")
            elif total_missing > 8:
                suggestions.append(f"⚠️ {total_missing} important skills missing. Significant gap in required qualifications.")
            elif total_missing > 0:
                suggestions.append(f"Consider adding these {total_missing} skills if you have experience with them.")
            
            sorted_missing = sorted(missing_skills.items(), key=lambda x: len(x[1]), reverse=True)
            for category, skills in sorted_missing[:3]:
                category_display = category.replace('_', ' ').title()
                skills_preview = ', '.join(skills[:8])
                if len(skills) > 8:
                    skills_preview += f" (+ {len(skills) - 8} more)"
                suggestions.append(f"Missing {category_display}: {skills_preview}")
        
        if scores['skills_match'] < 20:
            suggestions.append("Your technical skills don't align with this role's requirements.")
        
        resume_exp = self.extract_experience(resume_text)
        jd_exp = self.extract_experience(job_description)
        
        if jd_exp and resume_exp and resume_exp < jd_exp:
            suggestions.append(f"Job requires {jd_exp}+ years. Highlight relevant projects to demonstrate equivalent experience.")
        elif jd_exp and not resume_exp:
            suggestions.append(f"Add years of experience. Job requires {jd_exp}+ years.")
        
        required_sections = ['experience', 'education', 'skills', 'projects']
        missing_sections = [s for s in required_sections if s not in resume_text.lower()]
        
        if missing_sections:
            suggestions.append(f"Add missing sections: {', '.join(missing_sections).title()}")
        
        if scores['overall'] >= 40:
            action_verbs = ['developed', 'created', 'managed', 'led', 'implemented', 'designed', 'built', 'achieved', 'deployed', 'optimized']
            verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
            
            if verb_count < 5:
                suggestions.append("Use more action verbs: developed, implemented, managed, led, optimized")
        
        if scores['overall'] >= 40:
            numbers = re.findall(r'\d+%|\d+x|\d+\+', resume_text)
            if len(numbers) < 3:
                suggestions.append("Add quantifiable achievements (e.g., 'Improved performance by 40%')")
        
        if scores['overall'] >= 70 and len(suggestions) <= 2:
            suggestions.append("✅ Strong match! Consider tailoring further with exact job description phrases.")
        
        if not suggestions:
            suggestions.append("Review missing skills and job description carefully.")
        
        return suggestions
    
    def get_verdict(self, score):
        """Get hiring verdict based on score"""
        if score >= 80:
            return "✅ Excellent Match! High chance of passing ATS screening."
        elif score >= 60:
            return "✓ Good Match. Some improvements recommended to stand out."
        elif score >= 40:
            return "⚠️ Fair Match. Significant improvements needed to be competitive."
        else:
            return "❌ Poor Match. This role requires different skills and experience."
    
    def analyze(self, resume_text, job_description):
        """Main analysis function"""
        scores = self.calculate_ats_score(resume_text, job_description)
        resume_keywords = self.extract_keywords(resume_text, top_n=15)
        jd_keywords = self.extract_keywords(job_description, top_n=15)
        matching_keywords = list(set(resume_keywords) & set(jd_keywords))
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(job_description)
        missing_skills = self.find_missing_skills(resume_text, job_description)
        suggestions = self.generate_suggestions(resume_text, job_description, missing_skills, scores)
        
        return {
            'scores': scores,
            'resume_keywords': resume_keywords[:10],
            'jd_keywords': jd_keywords[:10],
            'matching_keywords': matching_keywords[:10],
            'resume_skills': resume_skills,
            'jd_skills': jd_skills,
            'missing_skills': missing_skills,
            'suggestions': suggestions,
            'verdict': self.get_verdict(scores['overall'])
        }


# Initialize analyzer
print("📄 Initializing Resume Analyzer...")
analyzer = ResumeAnalyzer()
print("✅ Resume Analyzer ready!")


def clean_extracted_text(text):
    """Clean and normalize extracted text"""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Common variations
    replacements = {
        'Scikit-learn': 'scikit-learn', 'ScikitLearn': 'scikit-learn', 'Scikit learn': 'scikit-learn',
        'TensorFlow': 'tensorflow', 'PyTorch': 'pytorch', 'NumPy': 'numpy', 'Pandas': 'pandas',
        'Node.js': 'node.js', 'Next.js': 'next.js', 'React.js': 'react', 'MongoDB': 'mongodb'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def extract_text_from_pdf(file_content):
    """Extract text from PDF"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = " ".join(page.extract_text() for page in pdf_reader.pages)
        return clean_extracted_text(text).strip()
    except Exception as e:
        raise Exception(f"Error parsing PDF: {str(e)}")


def extract_text_from_docx(file_content):
    """Extract text from DOCX"""
    try:
        docx_file = io.BytesIO(file_content)
        doc = docx.Document(docx_file)
        text = " ".join(paragraph.text for paragraph in doc.paragraphs)
        return clean_extracted_text(text).strip()
    except Exception as e:
        raise Exception(f"Error parsing DOCX: {str(e)}")


def extract_text_from_file(file_content, filename):
    """Extract text based on file extension"""
    ext = filename.lower()
    if ext.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif ext.endswith('.docx'):
        return extract_text_from_docx(file_content)
    elif ext.endswith('.txt'):
        return clean_extracted_text(file_content.decode('utf-8'))
    else:
        raise Exception("Unsupported file format. Upload PDF, DOCX, or TXT.")


@resume_bp.route('/')
def home():
    """Resume API info"""
    return jsonify({
        'status': 'running',
        'service': 'Resume Screener API',
        'version': '1.0.0',
        'features': ['ATS Scoring', 'Skills Matching', 'Keyword Analysis', 'Improvement Suggestions'],
        'supported_formats': ['PDF', 'DOCX', 'TXT'],
        'endpoints': {
            '/api/resume/': 'GET - API info',
            '/api/resume/analyze': 'POST - Analyze resume',
            '/api/resume/health': 'GET - Health check',
            '/api/resume/skills': 'GET - List supported skills'
        },
        'usage': {
            'method': 'POST',
            'endpoint': '/api/resume/analyze',
            'body': {
                'resume_file': 'base64_encoded_file',
                'resume_filename': 'resume.pdf',
                'job_description': 'Job description text...'
            }
        }
    })


@resume_bp.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'resume_screener',
        'analyzer_loaded': analyzer is not None,
        'spacy_loaded': nlp is not None
    })


@resume_bp.route('/skills')
def skills():
    """Get supported skills database"""
    return jsonify({
        'categories': list(analyzer.tech_skills.keys()),
        'total_skills': sum(len(skills) for skills in analyzer.tech_skills.values()),
        'skills_by_category': {
            category: len(skills) 
            for category, skills in analyzer.tech_skills.items()
        }
    })


@resume_bp.route('/analyze', methods=['POST'])
def analyze_resume():
    """Analyze resume against job description"""
    try:
        data = request.get_json()
        
        job_description = data.get('job_description', '')
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        resume_base64 = data.get('resume_file', '')
        resume_filename = data.get('resume_filename', '')
        
        if not resume_base64 or not resume_filename:
            return jsonify({'error': 'Resume file is required'}), 400
        
        # Decode base64
        try:
            if ',' in resume_base64:
                resume_base64 = resume_base64.split(',')[1]
            resume_content = base64.b64decode(resume_base64)
        except Exception as e:
            return jsonify({'error': f'Invalid file encoding: {str(e)}'}), 400
        
        # Extract text
        resume_text = extract_text_from_file(resume_content, resume_filename)
        
        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({
                'error': 'Could not extract enough text from resume. Please check the file.',
                'extracted_length': len(resume_text.strip())
            }), 400
        
        # Analyze
        result = analyzer.analyze(resume_text, job_description)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        print(f"❌ Resume analysis error: {str(e)}")
        return jsonify({
            'error': 'An error occurred during analysis',
            'details': str(e)
        }), 500
