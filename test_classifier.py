"""
Test the Sapphire embedding-centroid classifier.

Sends every example (futile + interessant) to Sapphire's /classify endpoint
and reports accuracy, misclassifications, and borderline cases.
"""

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

SAPPHIRE_URL = "http://127.0.0.1:3123/classify"
TIMEOUT = 10

EXPECTED_FUTILE = [
    # salutations
    "hello",
    "hi",
    "hi how are you",
    "what's up",
    "sup",
    "yo",
    "good morning",
    "good evening",
    "hey there",
    "howdy",
    "wassup",
    "heyy",
    "hii",
    "salut",
    "coucou",
    "salutations",
    "greetings",
    "how are you doing",
    "long time no see",
    # test du bot
    "are you a real person",
    "are you human",
    "good bot",
    "bad bot",
    "can you hear me",
    "is this thing on",
    "are you alive",
    "are you a bot",
    "what are you",
    "who made you",
    "do you have feelings",
    "can you think",
    # reactions creuses
    "lol",
    "haha nice",
    "haha ok",
    "lmao",
    "same energy fr",
    "lol same energy fr",
    "cool cool",
    "nice",
    "ok cool",
    "bet",
    "fair enough",
    "true true",
    "rip",
    "damn",
    "oof",
    "fr",
    "based",
    "real",
    "no way",
    "for real",
    "that's wild",
    "crazy",
    "wow",
    "lmfao",
    "lolz",
    "omg",
    "smh",
    "dead",
    "period",
    "facts",
    # ennui
    "bored af",
    "im bored",
    "nothing to do",
    "so bored rn",
    "im so bored",
    "kinda bored",
    "this is boring",
    "im dying of boredom",
    "entertain me",
    # divertissement
    "tell me a joke",
    "wanna play minecraft",
    "lets play a game",
    "truth or dare",
    "tell me a story",
    "guess what",
    "make me laugh",
    "say something funny",
    "tell me something interesting",
    # phatique
    "stop talking",
    "shut up",
    "go away",
    "leave me alone",
    "i love you",
    "you're the best",
    "thanks bot",
    "ty",
    "brb",
    "wyd",
    "k",
    "kk",
    "np",
    "nvm",
    "thanks",
    "thank you",
    "ok thanks",
    "tyvm",
    "appreciate it",
    "you too",
    "same",
    "mood",
    "big mood",
    "no problem",
    "my pleasure",
    "anytime",
    "sure thing",
    "gotcha",
    "i see",
    "ah ok",
    "makes sense",
    "alright",
    "aight",
    # questions creuses
    "what's your favorite color",
    "what's your name",
    "how old are you",
    "what day is it today",
    "what's the weather like",
    "do you like pizza",
    "do you like music",
    "what's your favorite movie",
    "do you have a pet",
    "what's your hobby",
    "pineapple on pizza",
    "are you a cat person or dog person",
    "what's your sign",
    "what's your favorite food",
    "do you like sports",
    "what's your birthday",
    "do you dream",
    "can you sing",
    "tell me a fun fact",
    # farewells
    "bye",
    "goodbye",
    "see you later",
    "gotta go",
    "good night",
    "take care",
    "peace out",
    "later",
    "catch you later",
    "ttyl",
    # affirmations
    "yes",
    "yeah",
    "okay",
    "alright",
    "sounds good",
    "absolutely",
    "agreed",
    "correct",
    "true",
    "for sure",
    # dismissive
    "i don't care",
    "whatever",
    "not my problem",
    "i'm indifferent",
    "pass",
    "no comment",
    "it is what it is",
    # off-topic
    "i like trains",
    "random",
    "side note",
    "fun fact",
    "i'm hungry",
    # confusion
    "i'm confused",
    "what do you mean",
    "i don't get it",
    "huh",
    "wait what",
    "can you repeat that",
    "explain like i'm five",
    # apologies
    "sorry",
    "my bad",
    "oops",
    "i apologize",
    "my mistake",
    # memes
    "this is fine",
    "bruh",
    "sus",
    "no cap",
    "yikes",
    "that's cap",
    "pog",
    "gg",
    # bot commands
    "/help",
    "/ping",
    "stop",
    "repeat",
    "remind me",
    # filler words
    "um",
    "uh",
    "so",
    "well",
    "i mean",
    "tbh",
    "ngl",
    "idk",
    "i dunno",
]

EXPECTED_INTERESSANT = [
    # technique / dev
    "i am a software engineer",
    "how do i deploy a cloudflare worker",
    "i need help with my code",
    "i'm building a react app and the state keeps resetting",
    "what's the difference between tcp and udp",
    "how does bitcoin mining work",
    "explain how neural networks work",
    "what's the best way to structure a rest api",
    "i just fixed a nasty memory leak",
    "how does garbage collection work in python",
    "i'm learning rust and the borrow checker is brutal",
    "how do i debug this docker container",
    "what's the difference between git merge and rebase",
    "how do i set up ci/cd for my project",
    "how do i handle authentication in a spa",
    "explain dependency injection",
    "what's the time complexity of this algorithm",
    "how does kubernetes service discovery work",
    "how do i use terraform with aws",
    "what's the best way to handle errors in go",
    "how does websocket work",
    "explain microservices architecture",
    "what's the difference between sql and nosql",
    "how do i implement caching in my app",
    # sciences
    "are you sentient",
    "do a magic trick",
    "what is quantum computing",
    "can you explain quantum physics",
    "how do vaccines actually work",
    "what's the difference between weather and climate",
    "why is the sky blue",
    "how do black holes form",
    "what's the theory of relativity",
    "how does photosynthesis work",
    "what causes earthquakes",
    "how does dna replication work",
    "explain the greenhouse effect",
    "how do batteries work",
    "what is dark matter",
    "how does evolution work",
    "what is the multiverse theory",
    "how do semiconductors work",
    "what causes volcanic eruptions",
    "how does the human brain work",
    # philosophie
    "what's the meaning of life",
    "do you think ai will take over",
    "i think capitalism is broken tbh",
    "what's your take on free will",
    "what's the difference between ethics and morality",
    "do you believe in fate",
    "is there life after death",
    "what is consciousness",
    "do animals have rights",
    "what is the nature of reality",
    "what is the best political system",
    "is there objective truth",
    "what makes a life meaningful",
    "do we have free will or is everything determined",
    # partage personnel
    "my girlfriend left me",
    "my dog just died",
    "i just got promoted at work",
    "i'm scared about my exam tomorrow",
    "i've been struggling with anxiety lately",
    "i lost my job today",
    "my parents are getting divorced",
    "i just moved to a new city and i feel lonely",
    "i finally finished my thesis",
    "i had a huge argument with my brother",
    "i think i'm falling in love",
    "i'm going through a tough time",
    "i don't know what to do with my life",
    "i just broke up with my partner",
    "i'm feeling really depressed lately",
    "my cat ran away",
    "i got diagnosed with something",
    "i'm really proud of my kid today",
    "my grandfather passed away",
    "i just adopted a puppy",
    "i'm struggling with my mental health",
    "i had a panic attack yesterday",
    # créativité
    "can you help me write a short story",
    "help me brainstorm a business idea",
    "can you recommend a book on stoicism",
    "what's a good strategy for learning a language",
    "how should i structure my resume",
    "how do i start investing",
    "what's the best way to prepare for an interview",
    "i want to start a youtube channel",
    "how do i learn to draw",
    "what should i write my novel about",
    "help me name my startup",
    "i need a slogan for my brand",
    "how do i compose music",
    "what camera should i buy for photography",
    "how do i start a podcast",
    # politique / société
    "thoughts on the israel palestine conflict",
    "what do you think about universal basic income",
    "is democracy the best system",
    "should we legalize all drugs",
    "what's your opinion on immigration",
    "why is healthcare so expensive",
    "thoughts on the housing crisis",
    "what caused the 2008 financial crisis",
    "explain the cold war",
    "what do you think about ai regulation",
    "is social media bad for society",
    "thoughts on income inequality",
    "what caused world war one",
    # demandes substantielles
    "can you help me with my homework",
    "i need advice about my relationship",
    "how do i tell my parents something difficult",
    "what career should i pursue",
    "i'm thinking about dropping out of school",
    "how do i make friends as an adult",
    "i need help planning my wedding",
    "how do i negotiate my salary",
    "i want to move abroad",
    "can you proofread my cover letter",
    "how do i deal with burnout",
    "how do i ask for a raise",
    "i need help with my resume",
    "how do i start my own business",
    "should i buy a house or rent",
    "how do i improve my public speaking",
    "i want to learn a new skill but i don't know where to start",
    # mental health
    "i think i need therapy",
    "my panic attacks are getting worse",
    "i can't stop overthinking",
    "i feel like nobody understands me",
    "i'm struggling with my mental health",
    "how do i cope with anxiety",
    "i feel so alone",
    "i'm having a really rough day",
    # relationship advice
    "my partner and i are fighting all the time",
    "how do i know if my relationship is healthy",
    "i caught my partner cheating",
    "how do i rebuild trust",
    "i need advice on setting boundaries",
    # career advice
    "i hate my job",
    "how do i switch careers",
    "i'm burnt out from work",
    "how do i negotiate a salary",
    "i was passed over for a promotion",
    # financial advice
    "i'm drowning in debt",
    "how do i start budgeting",
    "should i buy a house or rent",
    "how do i start investing",
    "my student loans are crushing me",
    # health
    "i've been having headaches every day",
    "i can't sleep at night",
    "what's the best diet for my condition",
    "how do i start working out",
    "i'm always tired",
    # educational
    "i want to learn python",
    "how do i write a research paper",
    "i need to pass an exam",
    "what's the best way to study",
    "i'm going back to school",
    # existential
    "why do we exist",
    "is there any meaning to suffering",
    "are we living in a simulation",
    "what happens when we die",
    "do we have free will",
    # ethical dilemmas
    "is it ever okay to lie",
    "should ai have rights",
    "is capitalism ethical",
    "is it wrong to cut off toxic family",
    # grief
    "i lost someone close to me",
    "how do i cope with grief",
    "i miss them so much",
    "grief comes in waves",
    # identity
    "how do i come out to my parents",
    "i'm questioning my identity",
    "i don't feel like i belong",
    "i'm proud of who i am",
    # spirituality
    "i'm exploring different religions",
    "how do i find inner peace",
    "i had a spiritual experience",
    "i want to start meditating",
    # parenting
    "i'm a new parent and i'm exhausted",
    "how do i discipline without yelling",
    "my teenager is rebelling",
    "i want to break the cycle of trauma",
    # addiction / recovery
    "i'm one year sober today",
    "how do i stay sober",
    "i relapsed and i feel like a failure",
    "i'm struggling with addiction",
    # self-improvement
    "how do i become more disciplined",
    "i want to stop procrastinating",
    "how do i build self esteem",
    "i want to be more productive",
    # world events
    "the climate crisis is terrifying",
    "what do you think about ai regulation",
    "inflation is destroying my savings",
    "thoughts on income inequality",
    # storytelling
    "let me tell you about the weirdest dream i had",
    "you won't believe what happened to me today",
    "the scariest thing that ever happened to me",
    "i once met someone who changed my perspective",
]


def classify(text: str) -> dict:
    req = Request(
        SAPPHIRE_URL,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def run(label_name: str, expected_label: str, examples: list[str]):
    ok = 0
    bad = 0
    borderline = []

    for text in examples:
        try:
            result = classify(text)
        except URLError as e:
            print(f"  ERROR connecting to Sapphire: {e}", file=sys.stderr)
            sys.exit(1)

        predicted = result["label"]
        conf = result["confidence"]
        sim_f = result["sim_futile"]
        sim_i = result["sim_interessant"]

        if predicted == expected_label:
            ok += 1
            if conf < 0.02:
                borderline.append((text, conf, sim_f, sim_i))
        else:
            bad += 1
            print(f"  MISMATCH [{predicted}] Δ={conf:.4f} f={sim_f:.4f} i={sim_i:.4f} | {text}")

    total = ok + bad
    pct = ok / total * 100 if total else 0
    print(f"\n{label_name}: {ok}/{total} ({pct:.1f}%) correct")
    if borderline:
        print(f"  borderline (|Δ| < 0.02, {len(borderline)}):")
        for text, conf, sf, si in borderline:
            print(f"    Δ={conf:.4f} f={sf:.4f} i={si:.4f} | {text}")
    return ok, bad


def main():
    print(f"Sapphire: {SAPPHIRE_URL}")
    print()

    if classify("hello").get("model") != "BAAI/bge-small-en-v1.5":
        print("  (checking via /classify result)")

    ok_f, bad_f = run("FUTILE", "FUTILE", EXPECTED_FUTILE)
    ok_i, bad_i = run("INTERESSANT", "INTERESSANT", EXPECTED_INTERESSANT)

    total = ok_f + ok_i + bad_f + bad_i
    correct = ok_f + ok_i
    print(f"\n=== TOTAL: {correct}/{total} ({correct/total*100:.1f}%) ===")


if __name__ == "__main__":
    main()
