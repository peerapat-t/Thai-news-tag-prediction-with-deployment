# %%
import warnings
warnings.simplefilter("ignore")

# %%
import pandas as pd
import numpy as np

# %%
import re
from pythainlp.util import normalize
from pythainlp import thai_characters
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
import pickle

# %%
import streamlit as st

# %% [markdown]
# # Clean text function

# %%
def normalize_text(text):
    return normalize(text)

# %%
def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# %%
def remove_urls(text):
    URL_PATTERN = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    return re.sub(URL_PATTERN, '', text)

# %%
def remove_non_thai_characters(text):
    allowed_characters = thai_characters
    escaped_allowed_characters = re.escape(allowed_characters)
    pattern = '[^' + escaped_allowed_characters + ']'
    return re.sub(pattern, '', text)

# %%
def remove_stopwords(text):
    stopwords = thai_stopwords()
    text_tokens = text.split()
    text_tokens = [word for word in text_tokens if word not in stopwords]
    return ''.join(text_tokens)

# %%
def remove_special_characters(text):
    pattern = r'[!@#$%^&*()\-+=\[\]{};:\'",<.>/?\\|]\n'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

# %%
def remove_words_from_text(text):
    words_to_remove = ['ประชาไท','ฯ','๑','๒','๓','๔','๕','๖','๗','๘','๙','๐']
    for word in words_to_remove:
        text = text.replace(word, '')
    return text

# %%
def remove_extra_spaces(text):
    return re.sub(r'\s+', '', text).strip()

# %%
def preprocess_text(text):
    text = normalize_text(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_non_thai_characters(text)
    text = remove_stopwords(text)
    text = remove_special_characters(text)
    text = remove_words_from_text(text)
    text = remove_extra_spaces(text)
    return word_tokenize(text, engine='newmm')

# %% [markdown]
# # Import trained model

# %%
politics_vectorization_model = pickle.load(open('vectorization model pickle/politics_vectorization.pickle', 'rb'))
human_rights_vectorization_model = pickle.load(open('vectorization model pickle/human_rights_vectorization.pickle', 'rb'))
quality_of_life_vectorization_model = pickle.load(open('vectorization model pickle/quality_of_life_vectorization.pickle', 'rb'))
foreign_affairs_vectorization_model = pickle.load(open('vectorization model pickle/foreign_affairs_vectorization.pickle', 'rb'))
society_vectorization_model = pickle.load(open('vectorization model pickle/society_vectorization.pickle', 'rb'))
environment_vectorization_model = pickle.load(open('vectorization model pickle/environment_vectorization.pickle', 'rb'))
economy_vectorization_model = pickle.load(open('vectorization model pickle/economy_vectorization.pickle', 'rb'))
culture_vectorization_model = pickle.load(open('vectorization model pickle/culture_vectorization.pickle', 'rb'))
labor_vectorization_model = pickle.load(open('vectorization model pickle/labor_vectorization.pickle', 'rb'))
security_vectorization_model = pickle.load(open('vectorization model pickle/security_vectorization.pickle', 'rb'))
ict_vectorization_model = pickle.load(open('vectorization model pickle/ict_vectorization.pickle', 'rb'))
education_vectorization_model = pickle.load(open('vectorization model pickle/education_vectorization.pickle', 'rb'))

# %%
politics_prediction_model = pickle.load(open('prediction model pickle/politics_prediction.pickle', 'rb'))
human_rights_prediction_model = pickle.load(open('prediction model pickle/human_rights_prediction.pickle', 'rb'))
quality_of_life_prediction_model = pickle.load(open('prediction model pickle/quality_of_life_prediction.pickle', 'rb'))
foreign_affairs_prediction_model = pickle.load(open('prediction model pickle/foreign_affairs_prediction.pickle', 'rb'))
society_prediction_model = pickle.load(open('prediction model pickle/society_prediction.pickle', 'rb'))
environment_prediction_model = pickle.load(open('prediction model pickle/environment_prediction.pickle', 'rb'))
economy_prediction_model = pickle.load(open('prediction model pickle/economy_prediction.pickle', 'rb'))
culture_prediction_model = pickle.load(open('prediction model pickle/culture_prediction.pickle', 'rb'))
labor_prediction_model = pickle.load(open('prediction model pickle/labor_prediction.pickle', 'rb'))
security_prediction_model = pickle.load(open('prediction model pickle/security_prediction.pickle', 'rb'))
ict_prediction_model = pickle.load(open('prediction model pickle/ict_prediction.pickle', 'rb'))
education_prediction_model = pickle.load(open('prediction model pickle/education_prediction.pickle', 'rb'))

# %% [markdown]
# # Prediction function

# %%
def prediction_model(title_input, body_input):

    result = []

    all_input = str(title_input) + str(body_input)

    # Politics prediction
    politics_vector = politics_vectorization_model.transform([all_input])
    politics_prediction = politics_prediction_model.predict(politics_vector)
    if politics_prediction[0] == 1:
        result.append('politics')
    else:
        pass

    # Human rights prediction
    human_rights_vector = human_rights_vectorization_model.transform([all_input])
    human_rights_prediction = human_rights_prediction_model.predict(human_rights_vector)
    if human_rights_prediction[0] == 1:
        result.append('human rights')
    else:
        pass


    # Quality of Life prediction
    quality_of_life_vector = quality_of_life_vectorization_model.transform([all_input])
    quality_of_life_prediction = quality_of_life_prediction_model.predict(quality_of_life_vector)
    if quality_of_life_prediction[0] == 1:
        result.append('quality of life')
    else:
        pass

    # Foreign Affairs prediction
    foreign_affairs_vector = foreign_affairs_vectorization_model.transform([all_input])
    foreign_affairs_prediction = foreign_affairs_prediction_model.predict(foreign_affairs_vector)
    if foreign_affairs_prediction[0] == 1:
        result.append('foreign affairs')
    else:
        pass

    # Society prediction
    society_vector = society_vectorization_model.transform([all_input])
    society_prediction = society_prediction_model.predict(society_vector)
    if society_prediction[0] == 1:
        result.append('society')
    else:
        pass

    # Environment prediction
    environment_vector = environment_vectorization_model.transform([all_input])
    environment_prediction = environment_prediction_model.predict(environment_vector)
    if environment_prediction[0] == 1:
        result.append('environment')
    else:
        pass

    # Economy prediction
    economy_vector = economy_vectorization_model.transform([all_input])
    economy_prediction = economy_prediction_model.predict(economy_vector)
    if economy_prediction[0] == 1:
        result.append('economy')
    else:
        pass

    # Culture prediction
    culture_vector = culture_vectorization_model.transform([all_input])
    culture_prediction = culture_prediction_model.predict(culture_vector)
    if culture_prediction[0] == 1:
        result.append('culture')
    else:
        pass

    # Labor prediction
    labor_vector = labor_vectorization_model.transform([all_input])
    labor_prediction = labor_prediction_model.predict(labor_vector)
    if labor_prediction[0] == 1:
        result.append('labor')
    else:
        pass

    # Security prediction
    security_vector = security_vectorization_model.transform([all_input])
    security_prediction = security_prediction_model.predict(security_vector)
    if security_prediction[0] == 1:
        result.append('security')
    else:
        pass

    # ICT prediction
    ict_vector = ict_vectorization_model.transform([all_input])
    ict_prediction = ict_prediction_model.predict(ict_vector)
    if ict_prediction[0] == 1:
        result.append('ICT')
    else:
        pass

    # Education prediction
    education_vector = education_vectorization_model.transform([all_input])
    education_prediction = education_prediction_model.predict(education_vector)
    if education_prediction[0] == 1:
        result.append('education')
    else:
        pass


    tag_string = ''

    if not result:
        tag_string = 'No tag'
    else:
        tag_string = ', '.join(result)

    return tag_string

# %% [markdown]
# # User interface

# %%
def main():
    st.title('Thai news tag prediction')

    title_input = st.text_input('News title')
    body_input = st.text_area('News body', height=300)

    tag_string = ''
    
    if st.button('Tag prediction !!'):
        tag_string = prediction_model(title_input, body_input)
        
    st.success(tag_string)

# %% [markdown]
# # Run

# %%
if __name__ == '__main__':
    main()


