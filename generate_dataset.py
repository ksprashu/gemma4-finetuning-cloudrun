import json
import random
import os

# Define categories and templates for generating high-quality, highly varied sentiment samples.
# We will generate exactly 2000 distinct samples across various domains, nuances, and linguistic edge cases.

# Domains to vary contexts
DOMAINS = ["product", "movie", "service", "food", "hotel", "software", "general", "book"]

# Sentiment classifications
POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"

# Data structures to hold seed words and phrases for combinatorics
templates = {
    # 1. Simple & Direct
    "simple_positive": [
        "This {domain} is absolutely wonderful.",
        "I am extremely satisfied with this {domain}.",
        "Simply the best {domain} I have ever encountered.",
        "Highly recommend this {domain} to anyone.",
        "An incredible {domain}, well worth the price.",
        "I love everything about this {domain}!",
        "Excellent quality and fantastic performance from this {domain}.",
        "This {domain} exceeded all my expectations."
    ],
    "simple_negative": [
        "This {domain} is terrible.",
        "I am very disappointed with this {domain}.",
        "Worst {domain} I have ever bought.",
        "Do not waste your money on this {domain}.",
        "Extremely poor quality and cheap build.",
        "I hate this {domain}, it does not work at all.",
        "Complete waste of time and effort.",
        "This {domain} failed within the first day of use."
    ],
    "simple_neutral": [
        "This {domain} is okay, nothing special.",
        "It is an average {domain} that does the job.",
        "The {domain} arrived on Tuesday.",
        "It is exactly as described in the manual.",
        "Not good, not bad, just standard.",
        "The {domain} has some basic features.",
        "I have been using this {domain} for two weeks now.",
        "It performs exactly as a basic {domain} should."
    ],
    
    # 2. Sarcasm & Irony (Edge case where surface words are positive but meaning is negative)
    "sarcastic_negative": [
        "Oh great, another {domain} that breaks instantly. Exactly what I wanted.",
        "Wow, a {domain} that lasts whole five minutes before dying. Impressive!",
        "Thanks for the user manual that is entirely in a language that doesn't exist.",
        "I love paying premium prices for a {domain} that works like a brick.",
        "Brilliant design! Who needs a working {domain} anyway?",
        "So glad I waited three weeks for this {domain} to arrive broken.",
        "It is so quiet because it doesn't turn on. Peaceful!",
        "Highly recommend this {domain} if you enjoy frustration and customer service hold music."
    ],
    
    # 3. Double Negation (Surface words look negative, but meaning is positive or neutral)
    "double_negation_positive": [
        "I don't dislike this {domain} at all.",
        "It's not that I'm unhappy with this {domain}.",
        "There is nothing bad to say about this {domain}.",
        "Not a bad {domain} by any stretch of the imagination.",
        "I cannot say that I am unsatisfied with this {domain}.",
        "This {domain} is not without its charm.",
        "Hardly a disappointment, to say the least.",
        "It doesn't fail to deliver on its promises."
    ],
    "double_negation_negative": [
        "I can't say it wasn't a mistake to buy this {domain}.",
        "It's not like it didn't fail to work.",
        "Not un-disappointed with this terrible {domain}.",
        "It is not uncommon for this {domain} to fail completely.",
        "No one could say this {domain} isn't a disaster.",
        "It did not not break on the first run."
    ],
    
    # 4. Mixed Sentiment & Contrast (But/However/Although clauses)
    "mixed_but_positive": [
        "The packaging of this {domain} was torn, but the actual performance is spectacular!",
        "Delivery took forever, however, the {domain} itself is perfect.",
        "Although it's a bit pricey, this {domain} is absolutely worth every single penny.",
        "It lacks a few advanced features, but it excels beautifully at what it does.",
        "The customer service was rude, but the {domain} is so good I don't even care.",
        "It has a steep learning curve, but once you get the hang of it, it's brilliant."
    ],
    "mixed_but_negative": [
        "The {domain} looks absolutely beautiful, but it functions terribly.",
        "It arrived super fast, however, the build quality is extremely disappointing.",
        "Although it has a lot of features, none of them actually work properly.",
        "The initial setup was incredibly easy, but the {domain} broke down after an hour.",
        "Customer service was nice, but they couldn't fix this fundamentally broken {domain}.",
        "It promised to revolutionize my workflow, but it only added frustration."
    ],
    
    # 5. Idiomatic Expressions & Slang (Modern contexts)
    "idiomatic_positive": [
        "This {domain} is absolutely top-tier!",
        "It hits the sweet spot perfectly.",
        "This {domain} is the absolute GOAT.",
        "I'm on cloud nine with this {domain}.",
        "This {domain} works like a charm.",
        "It blows the competition out of the water!",
        "This {domain} is a absolute game-changer.",
        "It's worth its weight in gold."
    ],
    "idiomatic_negative": [
        "This {domain} is a complete trainwreck.",
        "It really missed the mark this time.",
        "Buying this {domain} was throwing money down the drain.",
        "It is a total letdown.",
        "This {domain} is not up to par.",
        "It went down in flames on the first try.",
        "This {domain} is a wolf in sheep's clothing.",
        "It costs an arm and a leg for absolute garbage."
    ],
    
    # 6. Emojis and Emoticons only or heavy
    "emoji_positive": [
        "⭐⭐⭐⭐⭐ Best {domain} ever! 😍🙌",
        "This {domain} is 🔥🔥🔥",
        "💯 recommended! 👍👍👍",
        "Wow... just wow 🤩👌 {domain}",
        "Safe to say I am super happy with this 🥳🎉👏",
        "Perfect 💖✨"
    ],
    "emoji_negative": [
        "Worst ever 😡👎💔 {domain}",
        "Trash 🗑️💩🤢",
        "Broken on arrival... 😭😤",
        "Don't buy 🛑⚠️❌ {domain}",
        "Extremely disappointed 😔📉🤦‍♂️",
        "Useless piece of junk 🤬💀"
    ],
    "emoji_neutral": [
        "Received the {domain} today. 📦",
        "It's a standard {domain} 😐🤷‍♂️",
        "Just arrived 🗓️⏳",
        "Okay i guess 🧐",
        "The {domain} is black in color. 🖤",
        "Using it currently. 🔍"
    ],

    # 7. Typos and Grammatical errors
    "typos_positive": [
        "dis is so gud i lov it",
        "best {domain} evr bought!! highly recomended",
        "awsome quality nd super fast deliveryy",
        "i am so happyyyy with dis purchase",
        "works realy well no complains",
        "definitly worth d money!"
    ],
    "typos_negative": [
        "dis {domain} is so bad dnt buy",
        "worst {domain} evr, completely broken nd useless",
        "waste of mony totally disapointed",
        "it dont even turn on... so frustating",
        "cheap quality broke instently",
        "horible experience, do not recomend atal"
    ],

    # 8. Conditional Sentiment
    "conditional_positive": [
        "If you want something that lasts, this {domain} is the only real choice.",
        "Should you decide to invest in a {domain}, make sure it is this one.",
        "Unless you hate quality, you will love this {domain}.",
        "I would buy this {domain} again in a heartbeat if I ever needed another."
    ],
    "conditional_negative": [
        "I would like this {domain} only if it actually worked as advertised.",
        "If this is considered premium, I'd hate to see the budget option.",
        "Unless they fix the software bugs, this {domain} is a hard pass.",
        "I would have left a good review if the customer service wasn't so hostile."
    ],
    
    # 9. Technical & Domain-Specific Descriptions (Neutral)
    "domain_neutral": [
        "The {domain} features a dual-core processor and 8GB of system memory.",
        "It weighs approximately 1.5 kilograms and is constructed of matte plastic.",
        "The database schema for this {domain} requires three distinct indexes.",
        "This model is compatible with both 110V and 220V electrical outlets.",
        "The dimensions are 10 inches by 5 inches by 2 inches.",
        "The {domain} comes with a one-year standard warranty certificate."
    ],

    # 10. Long-winded Context-dependent reviews (Nuanced)
    "nuanced_positive": [
        "I was skeptical at first because of some negative reviews, and during the first couple of days of setting up the {domain}, I encountered a few minor issues with configuration. However, after updating the software and tweaking the settings, it works flawlessly and has completely changed how I perform daily tasks.",
        "We've tested several varieties of {domain} over the last five years in our office. While none of them are perfect, this particular model balances features, pricing, and longevity better than any other option on the market currently.",
        "Although the packaging could have been designed better to prevent scuffs during shipping, the actual item performs exactly as described. The controls are intuitive, the build is solid, and I cannot see myself going back to my old setup."
    ],
    "nuanced_negative": [
        "The marketing campaign made this {domain} look like a revolutionary breakthrough, and the design is admittedly beautiful. But once you start using it for actual work, you realize the design compromises functionality entirely, making it practically unusable for professional tasks.",
        "I really wanted to support this brand, and the initial unboxing experience was pleasant. Unfortunately, after only three days of normal, light usage, it started showing critical errors, and the company refused to offer a refund, citing fine print in the warranty.",
        "On paper, the specs of this {domain} are top of the line. In practice, however, thermal throttling is so severe that it runs slower than models half its price, rendering those impressive specs completely meaningless."
    ]
}

# Domain-specific word mappings for more natural programmatic variations
domain_lexicon = {
    "product": {
        "nouns": ["item", "device", "widget", "gadget", "equipment"],
        "adjectives": ["durable", "flimsy", "sleek", "clunky"]
    },
    "movie": {
        "nouns": ["film", "picture", "show", "screenplay", "adaptation"],
        "adjectives": ["cinematic", "boring", "masterful", "monotonous"]
    },
    "service": {
        "nouns": ["consultation", "support", "assistance", "billing", "process"],
        "adjectives": ["efficient", "sluggish", "courteous", "unprofessional"]
    },
    "food": {
        "nouns": ["meal", "dish", "cuisine", "recipe", "appetizer"],
        "adjectives": ["delicious", "bland", "savory", "stale"]
    },
    "hotel": {
        "nouns": ["room", "suite", "stay", "resort", "lodging"],
        "adjectives": ["luxurious", "cramped", "spotless", "dingy"]
    },
    "software": {
        "nouns": ["app", "program", "tool", "platform", "interface"],
        "adjectives": ["responsive", "buggy", "user-friendly", "bloated"]
    },
    "general": {
        "nouns": ["experience", "purchase", "acquisition", "option", "choice"],
        "adjectives": ["satisfying", "dreadful", "acceptable", "unsatisfactory"]
    },
    "book": {
        "nouns": ["novel", "read", "story", "thriller", "biography"],
        "adjectives": ["gripping", "slow-paced", "insightful", "cliché"]
    }
}

# Additional modifiers to increase randomness and variance
intensifiers = ["really", "truly", "absolutely", "exceptionally", "incredibly", "decidedly", "somewhat", "quite"]
punctuation_options = [".", "!", "!!", "...", "!?"]

def mutate_text(text, domain):
    """Replaces placeholders with domain-specific vocabulary and adds randomized mutations."""
    # Resolve the main {domain} placeholder
    lex = domain_lexicon[domain]
    chosen_noun = random.choice(lex["nouns"])
    mutated = text.replace("{domain}", chosen_noun)
    
    # Randomly inject intensifiers if there's a space before adjectives/verbs
    if random.random() < 0.2:
        for adj in ["wonderful", "satisfied", "best", "incredible", "terrible", "disappointed", "worst", "poor", "bad"]:
            if adj in mutated:
                mutated = mutated.replace(adj, f"{random.choice(intensifiers)} {adj}")
                break
                
    # Randomly vary trailing punctuation
    if mutated[-1] in [".", "!"]:
        mutated = mutated[:-1] + random.choice(punctuation_options)
        
    # Introduce occasional slight casing differences (mimicking real user text)
    if random.random() < 0.05:
        mutated = mutated.lower()
    elif random.random() < 0.02:
        mutated = mutated.upper()
        
    return mutated

def generate_dataset(num_samples=2000):
    samples = []
    generated_set = set() # For checking global uniqueness of prompts
    
    # Calculate target counts per category to ensure a highly balanced and rich distribution
    categories = list(templates.keys())
    
    # We want a diverse split across all template groups
    # Positive / Negative / Neutral balance: ~40% Positive, ~40% Negative, ~20% Neutral
    
    # Helper to determine sentiment of a template key
    def get_sentiment(key):
        if "positive" in key:
            return POSITIVE
        elif "negative" in key:
            return NEGATIVE
        else:
            return NEUTRAL
            
    while len(samples) < num_samples:
        # Pick a random category template
        cat = random.choice(categories)
        tpl_list = templates[cat]
        raw_tpl = random.choice(tpl_list)
        
        # Pick a domain
        domain = random.choice(DOMAINS)
        
        # Generate raw review text
        content_text = mutate_text(raw_tpl, domain)
        
        # Avoid exact duplicates in dataset
        if content_text in generated_set:
            continue
            
        generated_set.add(content_text)
        sentiment = get_sentiment(cat)
        
        # Formulate instruction structure for user
        # Variety in instructions as well to make training dynamic and prevent overfitting on specific phrasing
        instruction_formats = [
            "Classify the sentiment: '{text}'",
            "What is the sentiment of this text: '{text}'?",
            "Analyze the sentiment of the following text: '{text}'",
            "Sentiment classification for: '{text}'",
            "Determine the sentiment: '{text}'",
            "Review: '{text}' | Sentiment:"
        ]
        
        # Simple/direct instruction vs advanced instruction mapping
        instruction = random.choice(instruction_formats).format(text=content_text)
        
        # Create schema according to request
        sample = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "model", "content": sentiment}
            ]
        }
        
        samples.append(sample)
        
    return samples

def main():
    target_count = int(os.getenv("DATASET_SIZE", "2000"))
    print(f"Generating {target_count} highly varied sentiment analysis samples...")
    dataset = generate_dataset(target_count)
    
    # Verify dataset distribution
    pos_count = sum(1 for s in dataset if s["messages"][1]["content"] == POSITIVE)
    neg_count = sum(1 for s in dataset if s["messages"][1]["content"] == NEGATIVE)
    neu_count = sum(1 for s in dataset if s["messages"][1]["content"] == NEUTRAL)
    
    print("\nDataset Verification & Distribution:")
    print(f"Total Samples Generated: {len(dataset)}")
    print(f" - Positive: {pos_count} ({pos_count/len(dataset)*100:.1f}%)")
    print(f" - Negative: {neg_count} ({neg_count/len(dataset)*100:.1f}%)")
    print(f" - Neutral:  {neu_count} ({neu_count/len(dataset)*100:.1f}%)")
    
    # Resolve output file path from environment variable or fallback to default
    output_file = os.getenv("OUTPUT_FILE")
    if not output_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "sentiment_dataset.jsonl")
        
    print(f"\nWriting dataset to: {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Dataset created and saved successfully!")

if __name__ == "__main__":
    main()
