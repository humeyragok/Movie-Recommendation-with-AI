import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re
import json
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="🎬 Find a Movie",
    layout="wide",
    initial_sidebar_state="expanded"
)
def simple_theme():
    st.markdown("""
    <style>
    /* Sayfa genel ayarları */
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
        margin-top: -40px;
    }

    h1, h2, h3, h4 {
        color: #222222 !important;
        font-weight: 600;
    }

    /* Başlık alt çizgi */
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 15px 0;
    }

    /* Selectbox ve slider */
    .stSelectbox, .stSlider {
        background: white;
        padding: 8px;
        border-radius: 6px;
        border: 1px solid #ddd;
    }

    /* Slider yüksekliğini azalt */
    .stSlider {
        padding: 2px !important;
        min-height: 86px !important;
        height: 32px !important;
    }
    .stSlider .rc-slider {
        height: 18px !important;
        min-height: 68px !important;
    }
    .stSlider .rc-slider-rail, 
    .stSlider .rc-slider-track {
        height: 26px !important;
        min-height: 6px !important;
    }
    .stSlider .rc-slider-handle {
        width: 16px !important;
        height: 56px !important;
        margin-top: -5px !important;
        box-shadow: none !important;
    }

    /* Buton */
    div.stButton > button {
        background: #2E86C1;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 10px 16px;
        font-weight: 800;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #1B4F72;
        transform: scale(1.02);
    }

    /* Kart tasarımı */
    .stContainer {
        background: #ffffff;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .stContainer:hover {
        transform: translateY(-2px);
    }

    /* Metric kutuları */
    [data-testid="stMetric"] {
        background: #f1f1f1;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    """Veri dosyasını yükle"""
    try:
        data = pd.read_csv('tmdb-movie-metadata/tmdb_5000_movies.csv')
        return data
    except FileNotFoundError:
        return create_sample_data()
    except Exception as e:
        return create_sample_data()

def create_sample_data():
    """Örnek veri oluştur"""
    sample_data = {
        'title': [
            'The Matrix', 'Inception', 'Titanic', 'Avatar', 'The Godfather',
            'Pulp Fiction', 'The Dark Knight', 'Forrest Gump', 'Star Wars',
            'The Avengers', 'Jurassic Park', 'The Lion King', 'Frozen',
            'Finding Nemo', 'Toy Story', 'Shrek', 'Spider-Man', 'Iron Man',
            'Captain America', 'Thor', 'Interstellar', 'The Shawshank Redemption',
            'Gladiator', 'The Departed', 'Goodfellas'
        ],
        'genres': [
            'Action|Sci-Fi|Thriller', 'Action|Sci-Fi|Thriller', 'Drama|Romance',
            'Action|Adventure|Sci-Fi', 'Crime|Drama', 'Crime|Drama|Thriller',
            'Action|Crime|Drama', 'Drama|Romance', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Sci-Fi', 'Action|Adventure|Sci-Fi', 
            'Animation|Drama|Family', 'Animation|Adventure|Comedy',
            'Animation|Adventure|Comedy', 'Animation|Adventure|Comedy',
            'Animation|Adventure|Comedy', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Sci-Fi', 'Action|Adventure|Sci-Fi',
            'Action|Adventure|Fantasy', 'Drama|Sci-Fi|Thriller',
            'Drama|Crime', 'Action|Adventure|Drama', 'Crime|Drama|Thriller',
            'Crime|Drama|Biography'
        ],
        'overview': [
            'A computer hacker learns about the true nature of reality',
            'A thief who steals corporate secrets through dream-sharing technology',
            'A seventeen-year-old aristocrat falls in love with a penniless artist',
            'A paraplegic marine dispatched to the moon Pandora',
            'The aging patriarch of an organized crime dynasty',
            'The lives of two mob hitmen, a boxer and a gangster intertwine',
            'Batman must accept one of the greatest psychological tests',
            'The presidencies of Kennedy and Johnson through the eyes of an Alabama man',
            'Luke Skywalker joins forces with a Jedi Knight to rescue Princess Leia',
            'Earths mightiest heroes must come together to stop an alien invasion',
            'A pragmatic paleontologist visiting an almost complete theme park',
            'A Lion cub crown prince is tricked by his evil uncle',
            'When the newly crowned Queen Elsa accidentally uses her power',
            'After his son is captured in the Great Barrier Reef',
            'A cowboy doll is profoundly threatened by a new spaceman figure',
            'A mean lord exiles fairytale creatures to the swamp of Shrek',
            'When bitten by a genetically altered spider Peter Parker',
            'After being held captive in an Afghan cave billionaire engineer',
            'Steve Rogers a weak youth volunteers for a program that turns him',
            'The powerful but arrogant god Thor is cast out of Asgard',
            'A team of explorers travel through a wormhole in space',
            'Two imprisoned men bond over a number of years finding solace',
            'A former Roman General sets out to exact vengeance',
            'An undercover cop and a police informant discover their identities',
            'The story of Henry Hill and his life in the mob'
        ],
        'vote_average': [
            8.7, 8.8, 7.8, 7.8, 9.2, 8.9, 9.0, 8.8, 8.6, 8.0, 8.1, 8.5, 7.4, 8.2, 8.3, 7.9, 7.3, 7.9, 6.9, 7.0, 8.6, 9.3, 8.5, 8.5, 8.7
        ],
        'release_date': [
            '1999-03-31', '2010-07-16', '1997-12-19', '2009-12-18', '1972-03-24',
            '1994-10-14', '2008-07-18', '1994-07-06', '1977-05-25', '2012-05-04',
            '1993-06-11', '1994-06-24', '2013-11-27', '2003-05-30', '1995-11-22',
            '2001-05-18', '2002-05-03', '2008-05-02', '2011-07-22', '2011-05-06',
            '2014-11-07', '1994-09-23', '2000-05-05', '2006-10-06', '1990-09-21'
        ]
    }
    
    data = pd.DataFrame(sample_data)
    return data

def extract_genre_names(genres_data):
    """Tür verilerinden sadece isimleri çıkar"""
    if pd.isna(genres_data) or not genres_data:
        return "Belirtilmemiş"
    
    genres_str = str(genres_data)
    
    if genres_str.startswith('[') and 'name' in genres_str:
        try:

            genres_list = json.loads(genres_str)
            if isinstance(genres_list, list):
                names = [genre.get('name', '') for genre in genres_list if isinstance(genre, dict)]
                return ', '.join([name for name in names if name])
        except:
            pass
    

    if '|' in genres_str:
        genre_list = [genre.strip() for genre in genres_str.split('|')]
        return ', '.join([genre for genre in genre_list if genre])
    

    return genres_str.strip()

def clean_and_validate_data(data):
    """Veri temizleme ve doğrulama"""
    original_count = len(data)
    

    required_columns = ['title', 'genres']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"❌ Eksik sütunlar: {missing_columns}")
        st.stop()
    
    data = data.dropna(subset=['title', 'genres'])
    

    data = data[data['title'].str.strip() != '']
    data = data[data['genres'].str.strip() != '']
    

    if 'overview' not in data.columns:
        data['overview'] = 'No description available'
    else:
        data['overview'] = data['overview'].fillna('No description available')
    
    if 'vote_average' not in data.columns:
        data['vote_average'] = 7.0
    else:
        data['vote_average'] = pd.to_numeric(data['vote_average'], errors='coerce').fillna(7.0)
    
    cleaned_count = len(data)
    
    if cleaned_count == 0:
        st.error("❌ Temizleme sonrası veri kalmadı!")
        st.stop()
    
    return data.reset_index(drop=True)

def create_genre_features(data):
    """Tür özelliklerini oluştur"""
    all_genres = set()
    for genres in data['genres']:
        if pd.notna(genres) and genres.strip():
            genres_str = str(genres)
            
            if genres_str.startswith('[') and 'name' in genres_str:
                try:
                    genres_list = json.loads(genres_str)
                    if isinstance(genres_list, list):
                        for genre in genres_list:
                            if isinstance(genre, dict) and 'name' in genre:
                                all_genres.add(genre['name'].strip())
                except:

                    genre_list = [g.strip() for g in genres_str.split('|')]
                    all_genres.update(genre_list)
            else:
                genre_list = [g.strip() for g in genres_str.split('|')]
                all_genres.update(genre_list)
    
    all_genres = {g for g in all_genres if g and g.strip() and not g.startswith('[')}
    
    if not all_genres:
        st.error("❌ Hiç tür bulunamadı!")
        st.stop()
    
    for genre in sorted(all_genres):
        data[f'genre_{genre}'] = data['genres'].apply(
            lambda x: 1 if pd.notna(x) and genre in extract_genre_names(x) else 0
        )
    
    return data

def prepare_text_for_similarity(data):
    """Benzerlik analizi için metin hazırla"""
    def clean_text(text):
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        text = ' '.join(text.split())
        
        return text.lower()
    
    data['genres_clean'] = data['genres'].apply(lambda x: extract_genre_names(x).replace(',', ' ').lower() if pd.notna(x) else '')
    

    data['overview_clean'] = data['overview'].apply(clean_text)
    

    data['text_for_similarity'] = data.apply(
        lambda row: f"{row['genres_clean']} {row['overview_clean']}".strip(), 
        axis=1
    )
    

    empty_texts = data['text_for_similarity'].str.strip() == ''
    if empty_texts.any():
        data.loc[empty_texts, 'text_for_similarity'] = 'unknown movie'
    
    return data

@st.cache_data
def prepare_recommendation_system(data):
    """Öneri sistemini hazırla - cache'lenmiş"""
    try:

        data_clean = clean_and_validate_data(data.copy())
        

        data_with_genres = create_genre_features(data_clean)
        

        data_with_text = prepare_text_for_similarity(data_with_genres)
        

        try:
            tfidf = TfidfVectorizer(
                max_features=1000,
                min_df=1,  # En az 1 dokümanda geçmeli
                max_df=0.95,  # En fazla %95 dokümanda geçmeli
                ngram_range=(1, 2),  # 1-gram ve 2-gram
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            
            tfidf_matrix = tfidf.fit_transform(data_with_text['text_for_similarity'])
            
            if tfidf_matrix.shape[1] == 0:
                raise ValueError("TF-IDF matrisi boş")
            
        except Exception as tfidf_error:

            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(binary=True, lowercase=False)
            tfidf_matrix = vectorizer.fit_transform(data_with_text['genres_clean'])
        
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        return data_with_text, cosine_sim
        
    except Exception as e:
        st.error(f"❌ Öneri sistemi hazırlama hatası: {str(e)}")
        st.stop()

def get_recommendations(title, cosine_sim, data, num_recommendations=5):
    """Film önerilerini al"""
    try:
        title = title.strip()
        
        title_matches = data[data['title'].str.lower() == title.lower()]
        
        if title_matches.empty:
            partial_matches = data[data['title'].str.contains(title, case=False, na=False)]
            if not partial_matches.empty:
                title_matches = partial_matches.iloc[[0]]  
            else:
                return None
        
        idx = title_matches.index[0]
        

        sim_scores = list(enumerate(cosine_sim[idx]))
        

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        

        sim_scores = sim_scores[1:num_recommendations+1]
        

        movie_indices = [i[0] for i in sim_scores]
        

        recommendations = data.iloc[movie_indices][['title', 'genres', 'vote_average', 'overview']].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations
        
    except Exception as e:
        st.error(f"Öneri alma hatası: {str(e)}")
        return None

def display_movie_card(movie_data, similarity_score=None, is_selected=False):
    """Film kartı görüntüle"""
    with st.container():
        if is_selected:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%);
                padding: 40px;
                border-radius: 15px;
                margin: 20px 0;
                border: 3px solid #ff9800;
                box-shadow: 0 8px 25px rgba(255, 87, 34, 0.2);
                color: #222222;
            ">
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            poster_bg = "#2193b0" if is_selected else "#667eea"
            st.markdown(f"""
            <div style="
                width: 150px; 
                height: 225px; 
                background: linear-gradient(135deg, {poster_bg} 0%, #764ba2 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 48px;
                margin-bottom: 15px;
                box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            ">
                🎬
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Film başlığı
            if is_selected:
                st.markdown(f"""
                <h2 style='
                    color: #222222; 
                    text-shadow: none;
                    margin-bottom: 15px;
                    font-weight: bold;
                '>{movie_data['title']}</h2>
                """, unsafe_allow_html=True)
            else:
                st.subheader(movie_data['title'])
            
            # Benzerlik skoru
            if similarity_score:
                if is_selected:
                    st.markdown(f"""
                    <div style='
                        background: rgba(255,255,255,0.2);
                        padding: 10px;
                        border-radius: 8px;
                        margin-bottom: 15px;
                        color: #222222;
                        font-weight: bold;
                    '>
                        🎯 Benzerlik: {similarity_score:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("🎯 Benzerlik", f"{similarity_score:.1%}")
            
            # Puan ve türler
            col_rating, col_genre = st.columns(2)
            
            with col_rating:
                if is_selected:
                    st.markdown(f"""
                    <div style='
                        color: #222222; 
                        font-weight: bold;
                        text-shadow: none;
                        margin-bottom: 10px;
                    '>
                        ⭐ Puan: {movie_data['vote_average']:.1f}/10
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**⭐ Puan:** {movie_data['vote_average']:.1f}/10")
            
            with col_genre:
                genres = extract_genre_names(movie_data['genres'])
                if is_selected:
                    st.markdown(f"""
                    <div style='
                        color: #222222; 
                        font-weight: bold;
                        text-shadow: none;
                        margin-bottom: 10px;
                    '>
                        🎭 Türler: {genres}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**🎭 Türler:** {genres}")
            
            # Açıklama
            overview = str(movie_data['overview'])
            if len(overview) > 200:
                overview = overview[:200] + "..."
            
            if is_selected:
                st.markdown(f"""
                <div style='
                    color: #222222; 
                    font-weight: 500;
                    text-shadow: none;
                    margin-top: 15px;
                    line-height: 1.6;
                '>
                    📝 {overview}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"📝 {overview}")
        
        if is_selected:
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.title("🎬 Find a Movie") 
    st.markdown("---")
    

    # Ana içerik
    try:
        if 'data_loaded' not in st.session_state:
            data = load_data()
            if data is not None and not data.empty:
                data_processed, cosine_sim = prepare_recommendation_system(data)
                st.session_state.data_processed = data_processed
                st.session_state.cosine_sim = cosine_sim
                st.session_state.data_loaded = True
            else:
                st.error("❌ Veri yüklenemedi!")
                return
        
        data_processed = st.session_state.data_processed
        cosine_sim = st.session_state.cosine_sim
        
        st.header("🔍 Film Ara ve Öneri Al")
        
        movie_titles = sorted(data_processed['title'].unique())

        # Hepsi aynı satırda ve hizalı
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            selected_movie = st.selectbox(
                "Film seçin:",
                options=[""] + movie_titles,
                format_func=lambda x: "Film seçiniz..." if x == "" else x
            )
        with col2:
            num_recommendations = st.slider("Öneri Sayısı", 1, 10, 5, key="slider_recommendation_count")
        with col3:
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)  # Butonu aşağı almak için boşluk
            search_button = st.button("🎯 Öneri Al", type="primary", disabled=not selected_movie or selected_movie == "")

        if search_button and selected_movie and selected_movie != "":
            recommendations = get_recommendations(
                selected_movie, cosine_sim, data_processed, num_recommendations
            )
            
            if recommendations is not None and not recommendations.empty:
            
                st.subheader("🎬 Seçtiğiniz Film")
                selected_movie_data = data_processed[data_processed['title'] == selected_movie].iloc[0]
                display_movie_card(selected_movie_data, is_selected=True)
                
                st.markdown("---")
                
                st.subheader("🎯 Size Özel Öneriler")
                
                for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                    st.write(f"### {idx}. {movie['title']}")
                    display_movie_card(movie, movie['similarity_score'])
                    st.markdown("---")
            
            else:
                st.error("😔 Bu film için öneri bulunamadı. Başka bir film deneyin.")
    
    except Exception as e:
        st.error(f"❌ Uygulama hatası: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    simple_theme() 
    main()