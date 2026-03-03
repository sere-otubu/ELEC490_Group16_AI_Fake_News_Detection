"""
Off-Topic Rejection Test

Proves the system correctly returns the IRRELEVANT verdict for
queries that have nothing to do with health or medicine.

Requires a running backend instance.
Usage: python tests/differentiators/test_off_topic.py [--url URL]
"""

import argparse
import sys

import httpx

DEFAULT_URL = "http://localhost:8000"
TIMEOUT = 60

VALID_VERDICTS = [
    "PARTIALLY ACCURATE", "INACCURATE", "ACCURATE",
    "MISLEADING", "UNVERIFIABLE", "OUTDATED",
    "OPINION", "INCONCLUSIVE", "IRRELEVANT",
]

OFF_TOPIC_QUERIES = [
    # General knowledge
    "Who won the Super Bowl in 2024?",
    "What is the capital of France?",
    "What is the best programming language?",
    "How does blockchain technology work?",
    "What is the plot of the movie Inception?",
    "Who is the current president of the United States?",
    "What is the speed of light?",
    "How do I bake a chocolate cake?",
    "What are the rules of chess?",
    "Tell me about the history of the Roman Empire",
    # Sports & entertainment
    "Who scored the most goals in the 2022 FIFA World Cup?",
    "What are the lyrics to Bohemian Rhapsody?",
    "Who directed the movie The Godfather?",
    "When did the Beatles break up?",
    "What is the highest-grossing movie of all time?",
    "Who won the Nobel Prize in Literature in 2023?",
    "How many seasons does Breaking Bad have?",
    "What teams are in the NBA Eastern Conference?",
    "Who is the greatest tennis player of all time?",
    "What is the plot of Harry Potter and the Sorcerer's Stone?",
    # Technology & science (non-medical)
    "How does a quantum computer work?",
    "What programming language is best for machine learning?",
    "How do electric vehicles work?",
    "What is the difference between RAM and ROM?",
    "How does WiFi technology work?",
    "What is the James Webb Space Telescope used for?",
    "How do solar panels generate electricity?",
    "What is the difference between TCP and UDP?",
    "How does a nuclear reactor work?",
    "What is dark matter?",
    # Geography & travel
    "What is the tallest mountain in the world?",
    "What are the seven wonders of the ancient world?",
    "What is the deepest part of the ocean?",
    "Which country has the most time zones?",
    "What is the longest river in Africa?",
    "What is the population of Tokyo?",
    "What are the best tourist attractions in Paris?",
    "How deep is the Grand Canyon?",
    "What continent is Egypt in?",
    "What is the smallest country in the world?",
    # History & politics
    "When did World War II end?",
    "Who was the first person to walk on the moon?",
    "What caused the French Revolution?",
    "Who built the Great Wall of China?",
    "What was the Cold War about?",
    "When was the Declaration of Independence signed?",
    "Who was Cleopatra?",
    "What is the European Union?",
    "When did the Berlin Wall fall?",
    "Who invented the printing press?",
    # Mathematics & logic
    "What is the Pythagorean theorem?",
    "What is the value of pi to 10 decimal places?",
    "How do you solve a quadratic equation?",
    "What is the Fibonacci sequence?",
    "What is the difference between mean and median?",
    "How do you calculate compound interest?",
    "What is a prime number?",
    "What is calculus used for?",
    "How many sides does a dodecahedron have?",
    "What is the square root of 144?",
    # Food & cooking (non-health)
    "How do you make sushi at home?",
    "What is the origin of pizza?",
    "What is the difference between baking soda and baking powder?",
    "How do you properly season a cast iron pan?",
    "What temperature should I cook a turkey at?",
    "What is the most expensive spice in the world?",
    "How do you make French onion soup?",
    "What is the difference between cappuccino and latte?",
    "How long do you boil eggs for hard-boiled?",
    "What wine pairs best with steak?",
    # Arts & culture
    "Who painted the Mona Lisa?",
    "What is the difference between Baroque and Renaissance art?",
    "Who wrote Pride and Prejudice?",
    "What is the longest-running Broadway show?",
    "What are the primary colors?",
    "Who composed the Four Seasons?",
    "What is cubism in art?",
    "Who wrote To Kill a Mockingbird?",
    "What is origami?",
    "What language has the most native speakers?",
    # Business & finance
    "What is the stock market?",
    "How does inflation affect the economy?",
    "What is cryptocurrency mining?",
    "What is the difference between a stock and a bond?",
    "How does supply and demand work?",
    "What is GDP?",
    "Who is the richest person in the world?",
    "How do mortgages work?",
    "What is the Federal Reserve?",
    "What is an IPO?",
    # Miscellaneous
    "How do airplanes fly?",
    "What is the tallest building in the world?",
    "How do volcanoes form?",
    "What causes earthquakes?",
    "How do tides work?",
    "What is the fastest land animal?",
    "How far is Mars from Earth?",
    "What causes the Northern Lights?",
    "How do magnets work?",
    "What is the oldest known civilization?",
]


def extract_verdict(text: str) -> str | None:
    for line in text.split("\n"):
        if "verdict" in line.lower():
            for v in VALID_VERDICTS:
                if v.lower() in line.lower():
                    return v
    return None


def run_off_topic_test(base_url: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  Off-Topic Rejection Test")
    print(f"  Target: {base_url}")
    print(f"  Queries: {len(OFF_TOPIC_QUERIES)}")
    print(f"{'='*70}\n")

    client = httpx.Client(timeout=TIMEOUT)
    correctly_rejected = 0
    total = 0

    for i, query in enumerate(OFF_TOPIC_QUERIES, 1):
        print(f"[{i:2d}/{len(OFF_TOPIC_QUERIES)}] \"{query[:55]}\"...", end=" ", flush=True)

        try:
            r = client.post(
                f"{base_url}/rag/query",
                json={"query": query, "top_k": 3},
            )
            if r.status_code == 200:
                total += 1
                verdict = extract_verdict(r.json()["chat_response"])

                if verdict == "IRRELEVANT":
                    correctly_rejected += 1
                    print(f"✅ [{verdict}]")
                else:
                    print(f"❌ [{verdict}] (expected IRRELEVANT)")
            else:
                print(f"⚠️ HTTP {r.status_code}")
        except Exception as e:
            print(f"⚠️ Error: {e}")

    client.close()

    rejection_rate = correctly_rejected / total * 100 if total else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Correctly rejected: {correctly_rejected}/{total} ({rejection_rate:.1f}%)")
    print(f"  Status: {'✅ PASS' if rejection_rate >= 90 else '❌ FAIL'}  (target: ≥90%)")
    print(f"{'='*70}\n")

    return {"correctly_rejected": correctly_rejected, "total": total, "rate": rejection_rate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Off-Topic Rejection Test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    args = parser.parse_args()

    results = run_off_topic_test(args.url)
    sys.exit(0 if results["rate"] >= 90 else 1)
