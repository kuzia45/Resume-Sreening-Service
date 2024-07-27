import nltk
import streamlit as st
import pickle
import re



nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanTxt = re.sub('http\S+\s',' ', txt)
    cleanTxt = re.sub('RT|cc',' ', cleanTxt)
    cleanTxt = re.sub('@\S+',' ', cleanTxt)
    cleanTxt = re.sub('#\S+\s+',' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%@"()*+,-./:;<=>?@[\]^_`{|}~"""),' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]',' ', cleanTxt)
    cleanTxt = re.sub('\s+',' ', cleanTxt)
    return cleanTxt

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf', ])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        cleaned_resume = cleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]

        category_mapping = {6:'Data Science', 12:'HR', 0:'Advocate', 1:'Arts', 24:'Web Designing',
            16:'Mechanical Engineer', 22:'Sales', 14:'Health and fitness',
            5:'Civil Engineer', 15:'Java Developer', 4:'Business Analyst',
            21:'SAP Developer', 2:'Automation Testing', 11:'Electrical Engineering',
            18:'Operations Manager', 20:'Python Developer', 9:'DevOps Engineer',
            17:'Network Security Engineer', 19:'PMO', 7:'Database', 13:'Hadoop',
            10:'ETL Developer', 9:'DotNet Developer', 3:'Blockchain', 23:'Testing'}
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write('Predicted Category:', category_name)


if __name__=="__main__":
    main()
