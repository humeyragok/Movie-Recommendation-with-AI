# Film Öneri Sistemi - Jupyter Notebook İçin
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Matplotlib için Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

class MovieRecommendationSystem:
    def __init__(self, dataframe):
        """
        Film öneri sistemi sınıfı - DataFrame ile başlatılır
        """
        self.df = dataframe.copy()
        self.similarity_matrix = None
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.genre_columns = [col for col in self.df.columns if col.startswith('Tür_')]
        
        print(f"✅ {len(self.df)} film yüklendi!")
        print(f"📊 {len(self.genre_columns)} farklı tür bulundu")
        
    def explore_data(self):
        """
        Veri setini keşfet ve analiz et
        """
        print("\n" + "="*60)
        print("📊 VERİ SETİ ANALİZİ")
        print("="*60)
        
        print(f"Toplam film sayısı: {len(self.df)}")
        print(f"Sütun sayısı: {len(self.df.columns)}")
        
        print(f"\n🎬 Mevcut filmler:")
        for i, title in enumerate(self.df['title'], 1):
            vote_avg = self.df.iloc[i-1]['vote_average']
            vote_count = self.df.iloc[i-1]['vote_count']
            print(f"  {i}. {title} - ⭐{vote_avg}/10 ({vote_count:,} oy)")
        
        print(f"\n📈 İstatistiksel Özet:")
        print(f"  • Ortalama IMDB puanı: {self.df['vote_average'].mean():.2f}")
        print(f"  • En yüksek puan: {self.df['vote_average'].max()}")
        print(f"  • En düşük puan: {self.df['vote_average'].min()}")
        print(f"  • Ortalama oy sayısı: {self.df['vote_count'].mean():,.0f}")
        print(f"  • En çok oylanan: {self.df.loc[self.df['vote_count'].idxmax(), 'title']}")
        
        # Tür analizi
        if self.genre_columns:
            print(f"\n🎭 Tür Dağılımı:")
            genre_counts = self.df[self.genre_columns].sum().sort_values(ascending=False)
            for i, (genre, count) in enumerate(genre_counts.head(10).items(), 1):
                if count > 0:
                    print(f"  {i}. {genre.replace('Tür_', 'Tür ')}: {count} film")
        
        # Her filmin türlerini göster
        print(f"\n🎨 Film Türleri:")
        for i, row in self.df.iterrows():
            film_genres = []
            for genre_col in self.genre_columns:
                if row[genre_col] == 1:
                    film_genres.append(genre_col.replace('Tür_', 'Tür'))
            
            genre_text = ", ".join(film_genres) if film_genres else "Tür bilgisi yok"
            print(f"  • {row['title']}: {genre_text}")
        
        return self.df
    
    def visualize_data(self):
        """
        Veri görselleştirmesi
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🎬 Film Veri Seti Analizi', fontsize=16, fontweight='bold')
        
        # 1. IMDB Puanları
        axes[0,0].bar(self.df['title'], self.df['vote_average'], 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[0,0].set_title('📊 IMDB Puanları', fontweight='bold')
        axes[0,0].set_ylabel('IMDB Puanı')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Oy Sayıları
        axes[0,1].bar(self.df['title'], self.df['vote_count'], 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[0,1].set_title('🗳️ Oy Sayıları', fontweight='bold')
        axes[0,1].set_ylabel('Oy Sayısı')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Puan vs Oy Sayısı İlişkisi
        scatter = axes[1,0].scatter(self.df['vote_count'], self.df['vote_average'], 
                                   c=range(len(self.df)), cmap='viridis', s=100, alpha=0.7)
        axes[1,0].set_title('⭐ Puan vs Oy Sayısı İlişkisi', fontweight='bold')
        axes[1,0].set_xlabel('Oy Sayısı')
        axes[1,0].set_ylabel('IMDB Puanı')
        axes[1,0].grid(True, alpha=0.3)
        
        # Film isimlerini scatter plot'a ekle
        for i, txt in enumerate(self.df['title']):
            axes[1,0].annotate(txt[:10] + '...', 
                              (self.df['vote_count'].iloc[i], self.df['vote_average'].iloc[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Tür Dağılımı
        if self.genre_columns:
            genre_counts = self.df[self.genre_columns].sum().sort_values(ascending=False)
            top_genres = genre_counts[genre_counts > 0].head(8)
            
            if len(top_genres) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
                bars = axes[1,1].bar(range(len(top_genres)), top_genres.values, color=colors)
                axes[1,1].set_title('🎭 En Popüler Türler', fontweight='bold')
                axes[1,1].set_xlabel('Tür')
                axes[1,1].set_ylabel('Film Sayısı')
                axes[1,1].set_xticks(range(len(top_genres)))
                axes[1,1].set_xticklabels([g.replace('Tür_', '') for g in top_genres.index], rotation=45)
                axes[1,1].grid(True, alpha=0.3)
                
                # Bar'ların üzerine değer yaz
                for bar, value in zip(bars, top_genres.values):
                    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                                  int(value), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_features(self):
        """
        Öneri sistemi için özellik matrisi hazırla
        """
        print("\n" + "="*60)
        print("🔧 ÖZELLİK MATRİSİ HAZIRLANIYOR")
        print("="*60)
        
        # Numerik özellikler
        numeric_features = ['vote_average', 'vote_count']
        
        # Tüm özellikleri birleştir
        feature_columns = numeric_features + self.genre_columns
        self.feature_matrix = self.df[feature_columns].copy()
        
        print(f"📊 Kullanılan özellikler:")
        print(f"  • Numerik: {numeric_features}")
        print(f"  • Türler: {len(self.genre_columns)} adet")
        
        # Vote count'u normalize et (log transform)
        self.feature_matrix['vote_count'] = np.log1p(self.feature_matrix['vote_count'])
        print(f"  • Vote count log dönüşümü uygulandı")
        
        # Özellikleri standartlaştır
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"✅ Özellik matrisi boyutu: {self.feature_matrix_scaled.shape}")
        print("✅ Standartlaştırma tamamlandı!")
        
        return self.feature_matrix_scaled
    
    def calculate_similarity(self):
        """
        Film benzerlik matrisi hesapla
        """
        print("\n" + "="*50)
        print("🔍 BENZERLİK MATRİSİ HESAPLANIYOR")
        print("="*50)
        
        if not hasattr(self, 'feature_matrix_scaled'):
            self.prepare_features()
        
        # Cosine benzerlik hesapla
        self.similarity_matrix = cosine_similarity(self.feature_matrix_scaled)
        
        print(f"✅ Benzerlik matrisi boyutu: {self.similarity_matrix.shape}")
        
        # Benzerlik matrisini görselleştir
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.similarity_matrix, 
                   annot=True, 
                   fmt='.3f',
                   xticklabels=self.df['title'],
                   yticklabels=self.df['title'],
                   cmap='YlOrRd',
                   square=True)
        plt.title('🔥 Film Benzerlik Matrisi', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return self.similarity_matrix
    
    def get_recommendations(self, movie_title, num_recommendations=3):
        """
        Belirli bir film için öneriler getir
        """
        if self.similarity_matrix is None:
            self.calculate_similarity()
        
        # Film indexini bul
        try:
            movie_matches = self.df[self.df['title'].str.contains(movie_title, case=False, na=False)]
            if len(movie_matches) == 0:
                print(f"❌ '{movie_title}' filmi bulunamadı!")
                print("📽️ Mevcut filmler:")
                for i, title in enumerate(self.df['title'], 1):
                    print(f"  {i}. {title}")
                return None
            
            movie_idx = movie_matches.index[0]
            exact_title = movie_matches.iloc[0]['title']
            
        except IndexError:
            print(f"❌ '{movie_title}' filmi veri setinde bulunamadı!")
            return None
        
        # Benzerlik skorlarını al
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # En benzer filmleri al (ilk film kendisi olduğu için atla)
        similar_movies = sim_scores[1:num_recommendations+1]
        
        # Sonuçları hazırla
        recommendations = []
        for idx, score in similar_movies:
            movie_info = {
                'title': self.df.iloc[idx]['title'],
                'vote_average': self.df.iloc[idx]['vote_average'],
                'vote_count': self.df.iloc[idx]['vote_count'],
                'genres': self.get_movie_genres(idx),
                'similarity_score': round(score, 3)
            }
            recommendations.append(movie_info)
        
        return recommendations, exact_title
    
    def get_movie_genres(self, movie_idx):
        """
        Bir filmin türlerini getir
        """
        genres = []
        for genre_col in self.genre_columns:
            if self.df.iloc[movie_idx][genre_col] == 1:
                genres.append(genre_col.replace('Tür_', 'Tür'))
        return ", ".join(genres) if genres else "Tür bilgisi yok"
    
    def print_recommendations(self, movie_title, num_recommendations=3):
        """
        Önerileri güzel formatta yazdır
        """
        result = self.get_recommendations(movie_title, num_recommendations)
        
        if result is None:
            return
        
        recommendations, exact_title = result
        
        print(f"\n{'='*70}")
        print(f"🎯 '{exact_title}' FİLMİNE BENZER FİLMLER")
        print(f"{'='*70}")
        
        # Seçilen filmin bilgilerini göster
        selected_movie = self.df[self.df['title'] == exact_title].iloc[0]
        print(f"\n📽️ Seçilen Film:")
        print(f"   🎬 {exact_title}")
        print(f"   ⭐ IMDB: {selected_movie['vote_average']}/10 ({selected_movie['vote_count']:,} oy)")
        print(f"   🎭 Türler: {self.get_movie_genres(selected_movie.name)}")
        
        print(f"\n🔍 Benzer Filmler:")
        print("-" * 70)
        
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. 🎬 {movie['title']}")
            print(f"   ⭐ IMDB: {movie['vote_average']}/10 ({movie['vote_count']:,} oy)")
            print(f"   🎭 Türler: {movie['genres']}")
            print(f"   📊 Benzerlik Skoru: {movie['similarity_score']}")
            print("-" * 50)
        
        if len(recommendations) == 0:
            print("⚠️ Benzer film bulunamadı (çok az film var)")
    
    def get_all_similarities(self, movie_title):
        """
        Bir filme karşı tüm filmlerin benzerlik skorlarını göster
        """
        if self.similarity_matrix is None:
            self.calculate_similarity()
        
        try:
            movie_matches = self.df[self.df['title'].str.contains(movie_title, case=False, na=False)]
            movie_idx = movie_matches.index[0]
            exact_title = movie_matches.iloc[0]['title']
        except IndexError:
            print(f"❌ '{movie_title}' filmi bulunamadı!")
            return None
        
        print(f"\n🔍 '{exact_title}' filmine karşı tüm benzerlik skorları:")
        print("-" * 60)
        
        for i, other_title in enumerate(self.df['title']):
            if i != movie_idx:  # Kendisi hariç
                similarity = self.similarity_matrix[movie_idx][i]
                print(f"📽️ {other_title}: {similarity:.3f}")
    
    def compare_movies(self, movie1_title, movie2_title):
        """
        İki filmi karşılaştır
        """
        try:
            # İlk film
            movie1_matches = self.df[self.df['title'].str.contains(movie1_title, case=False, na=False)]
            movie1_idx = movie1_matches.index[0]
            movie1_info = movie1_matches.iloc[0]
            
            # İkinci film
            movie2_matches = self.df[self.df['title'].str.contains(movie2_title, case=False, na=False)]
            movie2_idx = movie2_matches.index[0]
            movie2_info = movie2_matches.iloc[0]
            
        except IndexError:
            print("❌ Bir veya iki film de bulunamadı!")
            return None
        
        if self.similarity_matrix is None:
            self.calculate_similarity()
        
        similarity = self.similarity_matrix[movie1_idx][movie2_idx]
        
        print(f"\n🆚 FİLM KARŞILAŞTIRMA")
        print("="*50)
        print(f"🎬 Film 1: {movie1_info['title']}")
        print(f"   ⭐ IMDB: {movie1_info['vote_average']}/10 ({movie1_info['vote_count']:,} oy)")
        print(f"   🎭 Türler: {self.get_movie_genres(movie1_idx)}")
        
        print(f"\n🎬 Film 2: {movie2_info['title']}")
        print(f"   ⭐ IMDB: {movie2_info['vote_average']}/10 ({movie2_info['vote_count']:,} oy)")
        print(f"   🎭 Türler: {self.get_movie_genres(movie2_idx)}")
        
        print(f"\n📊 Benzerlik Skoru: {similarity:.3f}")
        
        if similarity > 0.8:
            print("🔥 Çok benzer filmler!")
        elif similarity > 0.6:
            print("👍 Oldukça benzer filmler")
        elif similarity > 0.4:
            print("😐 Kısmen benzer filmler")
        else:
            print("❌ Çok farklı filmler")
        
        return similarity

# Jupyter Notebook'ta kullanım için fonksiyonlar
def load_and_analyze_movies():
    """
    Mevcut data değişkenini kullanarak film sistemini başlat
    """
    global movie_system
    
    try:
        # Jupyter'daki 'data' değişkenini kullan
        movie_system = MovieRecommendationSystem(data)
        return movie_system
    except NameError:
        print("❌ 'data' değişkeni bulunamadı!")
        print("💡 Önce şu kodu çalıştırın: data = pd.read_csv('your_file.csv')")
        return None

def quick_demo():
    """
    Hızlı demo fonksiyonu
    """
    if 'movie_system' not in globals():
        print("❌ Önce movie_system'i yükleyin: movie_system = load_and_analyze_movies()")
        return
    
    print("🎬 HİZLI DEMO BAŞLADI!")
    
    # Veri analizi
    movie_system.explore_data()
    
    # Görselleştirme
    movie_system.visualize_data()
    
    # Benzerlik hesaplama
    movie_system.calculate_similarity()
    
    # Örnek öneriler
    print("\n" + "🎯"*20 + " ÖNERİLER " + "🎯"*20)
    movie_system.print_recommendations('Avatar', 3)
    movie_system.print_recommendations('Spectre', 3)
    
    # Film karşılaştırması
    print("\n" + "🆚"*20 + " KARŞILAŞTIRMA " + "🆚"*20)
    movie_system.compare_movies('Avatar', 'John Carter')
    
    print("\n✅ Demo tamamlandı!")

# Jupyter Notebook için kolay kullanım
print("🎬 Film Öneri Sistemi Jupyter Notebook'a yüklendi!")
print("\n📋 Kullanım adımları:")
print("1. movie_system = load_and_analyze_movies()")
print("2. quick_demo()  # Hızlı demo için")
print("3. movie_system.print_recommendations('Film Adı', 3)")
print("4. movie_system.compare_movies('Film1', 'Film2')")