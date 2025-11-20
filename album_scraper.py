from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

class PitchforkSeleniumScraper:
    def __init__(self, headless=True):
        self.base_url = "https://pitchfork.com/reviews/albums/"
        self.reviews = []
        self.headless = headless
        self.driver = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10  # Stop after 10 consecutive failures
        self.should_stop = False
    
    def setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        return self.driver
    
    def close_driver(self):
        if self.driver:
            self.driver.quit()
    
    def get_review_links(self, num_reviews=12) -> List[str]:
        print(f"Fetching review links from {self.base_url}...")
        
        if not self.driver:
            self.setup_driver()
        
        links = set()
        page = 1
        max_pages = 50
        
        print(f"Navigating through pages to collect {num_reviews} reviews...")
        
        while len(links) < num_reviews and page <= max_pages:
            if page == 1:
                url = self.base_url
            else:
                url = f"{self.base_url}?page={page}"
            
            print(f"  Page {page}: Loading {url}")
            self.driver.get(url)
            time.sleep(1.5)
            
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            
            links_before = len(links)
            
            review_elements = soup.find_all('a', href=True)
            for element in review_elements:
                href = element['href']
                if '/reviews/albums/' in href and href.count('/') > 3:
                    full_url = f"https://pitchfork.com{href}" if not href.startswith('http') else href
                    if full_url != self.base_url:
                        links.add(full_url)
            
            new_links = len(links) - links_before
            print(f"    Found {new_links} new reviews (total: {len(links)})")
            
            if new_links == 0:
                print(f"  No more reviews found. Reached end at page {page}.")
                break
            
            if len(links) >= num_reviews:
                break
            
            page += 1
        
        if page > max_pages:
            print(f"  Reached max page limit ({max_pages}). Found {len(links)} reviews.")
        
        links_list = list(links)[:num_reviews]
        print(f"Returning {len(links_list)} review links\n")
        return links_list

    
    def scrape_review(self, url: str) -> Dict:
        if not self.driver:
            self.setup_driver()
        
        print(f"Scraping: {url}")
        
        self.driver.get(url)
        
        time.sleep(2)
        
        review_data = {
            'url': url,
            'album_name': '',
            'artist_name': '',
            'genre': '',
            'label': '',
            'release_year': '',
            'score': '',
            'review_text': ''
        }
        
        try:
            soup = BeautifulSoup(self.driver.page_source, 'lxml')
            
            album_tag = soup.find('h1')
            if album_tag:
                review_data['album_name'] = album_tag.get_text(strip=True)
            
            # Extract artist name
            artist_link = soup.find('a', href=lambda x: x and '/artists/' in x)
            if artist_link:
                review_data['artist_name'] = artist_link.get_text(strip=True)
            
            genre_found = False
            
            for p_tag in soup.find_all('p', class_=re.compile('InfoSliceValue')):
                text = p_tag.get_text(strip=True)
                if text and len(text) < 100:
                    prev = p_tag.find_previous_sibling('p')
                    if prev and 'Genre' in prev.get_text():
                        review_data['genre'] = text
                        genre_found = True
                        break
            
            if not genre_found:
                scripts = soup.find_all('script', type='application/ld+json')
                for script in scripts:
                    try:
                        import json as json_lib
                        data = json_lib.loads(script.string)
                        if isinstance(data, dict) and 'genre' in data:
                            review_data['genre'] = data['genre']
                            genre_found = True
                            break
                    except:
                        pass
            
            if not genre_found:
                html = self.driver.page_source
                genre_match = re.search(r'"genre"\s*:\s*"([^"]+)"', html)
                if genre_match:
                    genre_text = genre_match.group(1)
                    genre_text = genre_text.replace('\\u002F', '/')
                    review_data['genre'] = genre_text.strip()
            
            for p_tag in soup.find_all('p', class_=re.compile('InfoSliceValue')):
                text = p_tag.get_text(strip=True)
                if text and len(text) < 200:
                    prev = p_tag.find_previous_sibling('p')
                    if prev and 'Label' in prev.get_text():
                        review_data['label'] = text
                        break
            
            if not review_data['label']:
                html = self.driver.page_source
                label_match = re.search(r'"label"\s*:\s*"([^"]+)"', html)
                if label_match:
                    review_data['label'] = label_match.group(1).strip()
            
            for p_tag in soup.find_all('p', class_=re.compile('InfoSliceValue')):
                text = p_tag.get_text(strip=True)
                if text:
                    prev = p_tag.find_previous_sibling('p')
                    if prev and ('Released' in prev.get_text() or 'Release' in prev.get_text() or 'Year' in prev.get_text()):
                        review_data['release_year'] = text
                        break
            
            if not review_data['release_year']:
                html = self.driver.page_source
                year_match = re.search(r'"releaseYear"\s*:\s*"([^"]+)"', html)
                if year_match:
                    review_data['release_year'] = year_match.group(1).strip()
            
            for elem in soup.find_all(['p', 'span', 'div']):
                text = elem.get_text(strip=True)
                if re.match(r'^\d{1,2}\.\d$', text):
                    try:
                        score_val = float(text)
                        if 0 <= score_val <= 10:
                            review_data['score'] = text
                            break
                    except ValueError:
                        pass
            
            if not review_data['score']:
                html = self.driver.page_source
                score_patterns = [
                    r'score["\']?[:\s>]+(\d\.\d)',
                    r'rating["\']?[:\s>]+(\d\.\d)',
                    r'>\s*(\d\.\d)\s*<',
                ]
                for pattern in score_patterns:
                    matches = re.findall(pattern, html[:5000])
                    for match in matches:
                        try:
                            score_val = float(match)
                            if 0 <= score_val <= 10:
                                review_data['score'] = match
                                break
                        except ValueError:
                            continue
                    if review_data['score']:
                        break
            
            # Extract review text
            article_body = soup.find('article')
            if article_body:
                paragraphs = article_body.find_all('p')
                review_paragraphs = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 100]
                review_data['review_text'] = ' '.join(review_paragraphs)
            
            print(f"✓ Scraped: {review_data['artist_name']} - {review_data['album_name']} (Score: {review_data['score']}, Genre: {review_data['genre']}, Label: {review_data['label']}, Year: {review_data['release_year']})")
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        
        return review_data
    
    def scrape_multiple_reviews(self, num_reviews: int = 12, max_workers: int = 8) -> List[Dict]:
        print(f"Starting Selenium scrape of {num_reviews} reviews with {max_workers} workers...")
        
        try:
            review_links = self.get_review_links(num_reviews)
            print(f"Found {len(review_links)} review links\n")
            
            self.close_driver()
            
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {executor.submit(self._scrape_with_own_driver, link): link 
                                for link in review_links}
                
                for future in as_completed(future_to_url):
                    completed += 1
                    url = future_to_url[future]
                    try:
                        review = future.result()
                        if review:
                            # Check if it's a timeout error
                            if isinstance(review, dict) and review.get('error') == 'timeout':
                                self.consecutive_failures += 1
                                print(f"[{completed}/{len(review_links)}] ⚠️  Timeout ({self.consecutive_failures} consecutive)")
                                
                                # Stop if too many consecutive failures
                                if self.consecutive_failures >= self.max_consecutive_failures:
                                    print(f"\n⚠️  STOPPING: {self.max_consecutive_failures} consecutive timeouts/connection errors.")
                                    print(f"Saving {len(self.reviews)} successfully scraped reviews...\n")
                                    self.should_stop = True
                                    break
                            else:
                                # Successful scrape - reset consecutive failures
                                self.consecutive_failures = 0
                                self.reviews.append(review)
                                print(f"[{completed}/{len(review_links)}] ✓ Completed")
                        else:
                            # Other failure (not timeout)
                            print(f"[{completed}/{len(review_links)}] ✗ Failed: {url}")
                            # Don't count non-timeout failures as consecutive
                    except Exception as e:
                        print(f"[{completed}/{len(review_links)}] ✗ Error: {e}")
        
        except Exception as e:
            print(f"Error in parallel scraping: {e}")
        
        return self.reviews
    
    def _scrape_with_own_driver(self, url: str) -> Dict:
        driver = None
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set page load timeout
            driver.set_page_load_timeout(30)  # 30 seconds timeout
            
            driver.get(url)
            time.sleep(2)
            
            review_data = {
                'url': url,
                'album_name': '',
                'artist_name': '',
                'genre': '',
                'label': '',
                'release_year': '',
                'score': '',
                'review_text': ''
            }
            
            soup = BeautifulSoup(driver.page_source, 'lxml')
            
            album_tag = soup.find('h1')
            if album_tag:
                review_data['album_name'] = album_tag.get_text(strip=True)
            
            artist_link = soup.find('a', href=lambda x: x and '/artists/' in x)
            if artist_link:
                review_data['artist_name'] = artist_link.get_text(strip=True)
            
            for p_tag in soup.find_all('p', class_=re.compile('InfoSliceValue')):
                text = p_tag.get_text(strip=True)
                if text and len(text) < 100:
                    prev = p_tag.find_previous_sibling('p')
                    if prev and 'Genre' in prev.get_text():
                        review_data['genre'] = text
                        break
            
            if not review_data['genre']:
                html = driver.page_source
                genre_match = re.search(r'"genre"\s*:\s*"([^"]+)"', html)
                if genre_match:
                    genre_text = genre_match.group(1)
                    genre_text = genre_text.replace('\\u002F', '/')
                    review_data['genre'] = genre_text.strip()
            
            for p_tag in soup.find_all('p', class_=re.compile('InfoSliceValue')):
                text = p_tag.get_text(strip=True)
                if text and len(text) < 200:
                    prev = p_tag.find_previous_sibling('p')
                    if prev and 'Label' in prev.get_text():
                        review_data['label'] = text
                        break
            
            if not review_data['label']:
                html = driver.page_source
                label_match = re.search(r'"label"\s*:\s*"([^"]+)"', html)
                if label_match:
                    review_data['label'] = label_match.group(1).strip()
            
            if not review_data['release_year']:
                html = driver.page_source
                year_match = re.search(r'"releaseYear"\s*:\s*"([^"]+)"', html)
                if year_match:
                    review_data['release_year'] = year_match.group(1).strip()
            
            for elem in soup.find_all(['p', 'span', 'div']):
                text = elem.get_text(strip=True)
                if re.match(r'^\d{1,2}\.\d$', text):
                    try:
                        score_val = float(text)
                        if 0 <= score_val <= 10:
                            review_data['score'] = text
                            break
                    except ValueError:
                        pass
            
            if not review_data['score']:
                html = driver.page_source
                score_patterns = [
                    r'score["\']?[:\s>]+(\d\.\d)',
                    r'rating["\']?[:\s>]+(\d\.\d)',
                    r'>\s*(\d\.\d)\s*<',
                ]
                for pattern in score_patterns:
                    matches = re.findall(pattern, html[:5000])
                    for match in matches:
                        try:
                            score_val = float(match)
                            if 0 <= score_val <= 10:
                                review_data['score'] = match
                                break
                        except ValueError:
                            continue
                    if review_data['score']:
                        break
            
            article_body = soup.find('article')
            if article_body:
                paragraphs = article_body.find_all('p')
                review_paragraphs = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 100]
                review_data['review_text'] = ' '.join(review_paragraphs)
            
            return review_data
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a timeout or connection error
            is_timeout = any(keyword in error_msg.lower() for keyword in 
                           ['timeout', 'timed out', 'connection', 'remote disconnected', 'max retries'])
            
            if is_timeout:
                print(f"⚠️  Timeout/Connection error for {url}: {e}")
                return {'error': 'timeout'}  # Special indicator for timeout
            else:
                print(f"Error scraping {url}: {e}")
                return None
        finally:
            if driver:
                driver.quit()
    
    def save_to_csv(self, filename: str = 'outputs/pitchfork_reviews.csv'):
        if not self.reviews:
            print("No reviews to save")
            return
        
        import os
        os.makedirs('outputs', exist_ok=True)
        
        keys = self.reviews[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.reviews)
        print(f"✓ Saved {len(self.reviews)} reviews to {filename}")


def main():
    scraper = PitchforkSeleniumScraper(headless=True)
    
    try:
        reviews = scraper.scrape_multiple_reviews(num_reviews=5000)
        
        # Always save what we have, even if we stopped early
        if reviews:
            scraper.save_to_csv()
            
            print("\n" + "="*60)
            print(f"SCRAPING SUMMARY: {len(reviews)} reviews collected")
            print("="*60)
            
            if scraper.should_stop:
                print("⚠️  Scraping stopped early due to connection issues")
                print(f"✓ Saved partial dataset with {len(reviews)} reviews\n")
            
            sample = reviews[0]
            print("Sample Review:")
            print(f"Album: {sample['album_name']}")
            print(f"Artist: {sample['artist_name']}")
            print(f"Genre: {sample['genre']}")
            print(f"Label: {sample['label']}")
            print(f"Release Year: {sample['release_year']}")
            print(f"Score: {sample['score']}")
            print(f"Review excerpt: {sample['review_text'][:200]}...")
        else:
            print("No reviews were scraped. Check the selectors and page structure.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        if scraper.reviews:
            print(f"Saving {len(scraper.reviews)} reviews collected so far...")
            scraper.save_to_csv()
            print("✓ Partial dataset saved\n")
        scraper.close_driver()
    except Exception as e:
        print(f"Error during scraping: {e}")
        if scraper.reviews:
            print(f"Saving {len(scraper.reviews)} reviews collected before error...")
            scraper.save_to_csv()
        scraper.close_driver()

if __name__ == "__main__":
    main()
