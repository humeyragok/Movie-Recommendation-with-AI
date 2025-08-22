import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Test verisi yükle veya oluştur"""
    try:

        data = pd.read_csv('movies.csv')
        print(f"✓ Veri yüklendi: {len(data)} film")
        return data
    except FileNotFoundError:
        print("⚠ movies.csv bulunamadı, örnek veri oluşturuluyor...")
        # Örnek test verisi 
        sample_data = {
            'title': [
                'The Matrix', 'Inception', 'Titanic', 'Avatar', 'The Godfather',
                'Pulp Fiction', 'The Dark Knight', 'Forrest Gump', 'Star Wars',
                'The Avengers', 'Jurassic Park', 'The Lion King', 'Frozen',
                'Finding Nemo', 'Toy Story', 'Shrek', 'Spider-Man', 'Iron Man',
                'Captain America', 'Thor'
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
                'Action|Adventure|Fantasy'
            ],
            'overview': [
                'A hacker discovers reality is a simulation',
                'Dreams within dreams heist movie',
                'Love story on a sinking ship',
                'Alien world with blue creatures',
                'Mafia family saga',
                'Interconnected crime stories',
                'Batman fights the Joker',
                'Life story of a simple man',
                'Space opera adventure',
                'Superheroes team up',
                'Dinosaurs come back to life',
                'Lion cub becomes king',
                'Princess with ice powers',
                'Lost fish adventure',
                'Toys come to life',
                'Ogre and princess love story',
                'Superhero with spider powers',
                'Genius builds armor suit',
                'Super soldier in WW2',
                'God of thunder'
            ],
            'vote_average': [8.7, 8.8, 7.8, 7.8, 9.2, 8.9, 9.0, 8.8, 8.6, 8.0, 8.1, 8.5, 7.4, 8.2, 8.3, 7.9, 7.3, 7.9, 6.9, 7.0],
            'release_date': [
                '1999-03-31', '2010-07-16', '1997-12-19', '2009-12-18', '1972-03-24',
                '1994-10-14', '2008-07-18', '1994-07-06', '1977-05-25', '2012-05-04',
                '1993-06-11', '1994-06-24', '2013-11-27', '2003-05-30', '1995-11-22',
                '2001-05-18', '2002-05-03', '2008-05-02', '2011-07-22', '2011-05-06'
            ]
        }
        data = pd.DataFrame(sample_data)
        print(f"✓ Örnek veri oluşturuldu: {len(data)} film")
        return data

def create_genre_features_test_fixed(data):
    """Tür özelliklerini oluştur - düzeltilmiş versiyon"""
    print("\n=== DÜZELTİLMİŞ TÜR TESTİ ===")
    
    # Tüm türleri çıkar
    all_genres = set()
    for genres in data['genres']:
        if pd.notna(genres):
            genre_list = genres.split('|')
            all_genres.update(genre_list)
    
    print(f"Bulunan türler: {sorted(all_genres)}")
    
    # Her tür için binary özellik oluştur
    for genre in sorted(all_genres):
        data[f'genre_{genre}'] = data['genres'].apply(
            lambda x: 1 if pd.notna(x) and genre in x.split('|') else 0
        )
    
    print(f"✓ {len(all_genres)} tür özelliği oluşturuldu")
    return data

def test_content_based_filtering(data):
    """İçerik tabanlı filtreleme testi"""
    print("\n=== İÇERİK TABANLI FİLTRELEME TESTİ ===")
    
    data['combined_features'] = data['genres'].fillna('') + ' ' + data['overview'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    print(f"✓ TF-IDF matrisi oluşturuldu: {tfidf_matrix.shape}")
    print(f"✓ Cosine similarity matrisi oluşturuldu: {cosine_sim.shape}")
    
    return cosine_sim, data

def get_recommendations(title, cosine_sim, data, num_recommendations=5):
    """Film önerisi al"""
    try:
        idx = data[data['title'] == title].index[0]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        sim_scores = sim_scores[1:num_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = data.iloc[movie_indices][['title', 'genres', 'vote_average']]
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations
        
    except IndexError:
        print(f"⚠ '{title}' filmi bulunamadı!")
        return None

def test_recommendations():
    """Öneri sistemini test et"""
    print("\n=== ÖNERİ SİSTEMİ TESTİ ===")
    
    data = load_data()
    
    data = create_genre_features_test_fixed(data)
    
    cosine_sim, data = test_content_based_filtering(data)
    
    test_movies = ['The Matrix', 'Titanic', 'The Lion King']
    
    for movie in test_movies:
        if movie in data['title'].values:
            print(f"\n'{movie}' filmi için öneriler:")
            recommendations = get_recommendations(movie, cosine_sim, data)
            if recommendations is not None:
                print(recommendations.to_string(index=False))
        else:
            print(f"⚠ '{movie}' filmi veri setinde bulunamadı")

def test_genre_distribution():
    """Tür dağılımını test et"""
    print("\n=== TÜR DAĞILIMI ANALİZİ ===")
    
    data = load_data()
    
    genre_counts = {}
    for genres in data['genres']:
        if pd.notna(genres):
            for genre in genres.split('|'):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("En popüler türler:")
    for genre, count in sorted_genres[:10]:
        print(f"  {genre}: {count} film")

def test_rating_distribution():
    """Puan dağılımını test et"""
    print("\n=== PUAN DAĞILIMI ANALİZİ ===")
    
    data = load_data()
    
    print(f"Ortalama puan: {data['vote_average'].mean():.2f}")
    print(f"En yüksek puan: {data['vote_average'].max():.2f}")
    print(f"En düşük puan: {data['vote_average'].min():.2f}")
    
    top_movies = data.nlargest(5, 'vote_average')[['title', 'vote_average', 'genres']]
    print("\nEn yüksek puanlı 5 film:")
    print(top_movies.to_string(index=False))

def main():
    """Ana test fonksiyonu"""
    print("🎬 FİLM ÖNERİ SİSTEMİ TESTLERİ")
    print("=" * 50)
    
    try:
        # Tüm testleri çalıştır
        test_genre_distribution()
        test_rating_distribution()
        test_recommendations()
        
        print("\n✅ Tüm testler başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()