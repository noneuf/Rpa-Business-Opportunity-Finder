import os
import requests
import json
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
import re

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

visited_urls = set()

def is_valid_internal_link(base_url, link):
    if not link or link.startswith(('javascript:', '#')):
        return False
    parsed_base = urlparse(base_url)
    parsed_link = urlparse(link)
    return parsed_link.netloc == "" or parsed_link.netloc == parsed_base.netloc

def crawl_website(start_url, max_depth=2):
    def crawl(url, depth):
        if depth > max_depth or url in visited_urls:
            return
        visited_urls.add(url)
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all("a", href=True)
            for tag in links:
                href = tag.get("href")
                full_url = urljoin(url, href)
                if is_valid_internal_link(start_url, full_url):
                    crawl(full_url, depth + 1)
        except Exception as e:
            print(f"[ERROR] Failed to crawl {url}: {e}")

    crawl(start_url, 0)
    return list(visited_urls)

async def get_preview(url, max_chars=800):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                for tag in soup(["script", "style", "img", "input"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                return text[:max_chars]
    except:
        return ""

async def filter_relevant_urls_with_openai(urls, batch_size=8, prompt=None, model="gpt-4o", on_disconnect=None):
    all_structured_results = []

    default_prompt_intro = (
        "For each of the following webpages, determine if it describes a business process or workflow that could be automated with RPA. If yes, summarize the RPA opportunity in one sentence.\n"
        "Respond ONLY with a JSON array using this exact format:\n"
        "[{\"url\": \"...\", \"summary\": \"...\", \"rpa_opportunity\": \"...\"}]\n\n"
    )

    for i in range(0, len(urls), batch_size):
        if on_disconnect and await on_disconnect():
            print("❌ Disconnected before batch started")
            break

        batch = urls[i:i+batch_size]
        print(f"\n[INFO] Processing batch {i // batch_size + 1} ({len(batch)} URLs)...")
        previews = []

        for url in batch:
            preview = await get_preview(url)
            if preview:
                previews.append(f"URL: {url}\nPreview: {preview}")

        if not previews:
            print("[WARNING] No previews found in this batch. Skipping.")
            continue

        full_prompt = (prompt or default_prompt_intro) + "\n\n---\n\n".join(previews)

        try:
            print("[ACTION] Sending to OpenAI...")
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            result_text = response.choices[0].message.content
            print(f"[SUCCESS] Batch {i // batch_size + 1} complete.\n")

            try:
                match = re.search(r'\[\s*{.*?}\s*\]', result_text, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    parsed = json.loads(json_text)
                    if isinstance(parsed, list):
                        for item in parsed:
                            opportunity = item.get("rpa_opportunity", "").lower()
                            if opportunity and not any(keyword in opportunity for keyword in ["no rpa", "none", "n/a", "not identified", "informational"]):
                                all_structured_results.append(item)
                    else:
                        print("[WARNING] Parsed result is not a list")
                else:
                    print("[WARNING] No JSON array found in result:")
                    print(result_text)
            except Exception as e:
                print(f"[ERROR] Failed to parse OpenAI response: {e}")

        except Exception as e:
            print(f"[ERROR] Batch {i//batch_size + 1} failed: {str(e)}")

        if on_disconnect and await on_disconnect():
            print("❌ Disconnected after batch, stopping early")
            break

    return all_structured_results
