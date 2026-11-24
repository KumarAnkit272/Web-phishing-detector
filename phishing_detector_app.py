import math
import re
import tkinter as tk
from tkinter import ttk, messagebox
from urllib.parse import urlparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =====================================================================
# 1. LEXICAL & STRUCTURAL FEATURE EXTRACTION ENGINE
# =====================================================================
class URLFeatureExtractor:
    """Extracts 16 lexical, statistical, and structural features from a URL."""

    SUSPICIOUS_TLDS = {
        '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.pw',
        '.club', '.work', '.buzz', '.info', '.click', '.loan', '.zip'
    }

    SHORTENING_SERVICES = (
        r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
        r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
        r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
        r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.ly|rb\.gy"
    )

    SENSITIVE_KEYWORDS = [
        'login', 'signin', 'verify', 'verification', 'account', 'update',
        'secure', 'banking', 'paypal', 'password', 'confirm', 'wallet',
        'support', 'billing', 'unlock', 'suspend', 'kyc', 'recover', 'auth'
    ]

    FEATURE_NAMES = [
        "URL Length",
        "Domain Length",
        "Has IP Address",
        "Has '@' Symbol",
        "Double Slash Redirect",
        "Hyphen in Domain",
        "Subdomain Count",
        "HTTPS Protocol",
        "Fake 'https' in Domain",
        "Dot Count",
        "Digit Ratio",
        "Special Char Count",
        "Sensitive Keywords",
        "Suspicious TLD",
        "URL Shortener",
        "Shannon Entropy"
    ]

    @staticmethod
    def calculate_entropy(text: str) -> float:
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
        return round(entropy, 3)

    @classmethod
    def extract_features(cls, url: str) -> list:
        url = url.strip()
        # Ensure scheme exists for accurate urlparse netloc extraction
        parse_target = url if re.match(r'^https?://', url, re.I) else 'http://' + url
        parsed = urlparse(parse_target)

        scheme = parsed.scheme.lower()
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        # Strip port if present
        domain_clean = domain.split(':')[0]
        full_lower = url.lower()

        # 1. URL Length
        url_len = len(url)

        # 2. Domain Length
        domain_len = len(domain_clean)

        # 3. Has IP Address (IPv4 or hex IP in domain)
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        has_ip = 1 if re.match(ip_pattern, domain_clean) else 0

        # 4. Has '@' Symbol
        has_at = 1 if '@' in url else 0

        # 5. Double Slash Redirect after protocol
        pos_double_slash = url.rfind('//')
        double_slash_redirect = 1 if pos_double_slash > 7 else 0

        # 6. Hyphen in Domain
        has_hyphen = 1 if '-' in domain_clean else 0

        # 7. Subdomain Count
        dots_in_domain = domain_clean.count('.')
        subdomain_count = max(0, dots_in_domain - 1) if not has_ip else 0

        # 8. Uses HTTPS Protocol (explicitly provided by user)
        is_https = 1 if url.lower().startswith('https://') else 0

        # 9. Fake 'https' or 'http' inside the domain name itself
        fake_https = 1 if ('https' in domain_clean or 'http' in domain_clean) else 0

        # 10. Total Dot Count in URL
        dot_count = url.count('.')

        # 11. Digit Ratio in URL
        digits = sum(c.isdigit() for c in url)
        digit_ratio = round(digits / max(1, len(url)), 3)

        # 12. Special Character Count
        special_chars = sum(url.count(c) for c in ['-', '_', '?', '=', '&', '%', '$', '!'])

        # 13. Sensitive Keywords Count
        keyword_count = sum(1 for kw in cls.SENSITIVE_KEYWORDS if kw in full_lower)

        # 14. Suspicious TLD Check
        suspicious_tld = 1 if any(domain_clean.endswith(tld) for tld in cls.SUSPICIOUS_TLDS) else 0

        # 15. Shortening Service Check
        is_shortener = 1 if re.search(cls.SHORTENING_SERVICES, domain_clean, re.I) else 0

        # 16. Shannon Entropy (measures randomness of URL characters)
        entropy = cls.calculate_entropy(url)

        return [
            url_len, domain_len, has_ip, has_at, double_slash_redirect,
            has_hyphen, subdomain_count, is_https, fake_https, dot_count,
            digit_ratio, special_chars, keyword_count, suspicious_tld,
            is_shortener, entropy
        ]

    @classmethod
    def inspect_risk_factors(cls, url: str, features: list) -> list:
        """Generates human-readable explanations for each extracted feature."""
        explanations = []
        (url_len, domain_len, has_ip, has_at, double_slash,
         has_hyphen, subdomains, is_https, fake_https, dot_count,
         digit_ratio, special_chars, kw_count, susp_tld,
         is_shortener, entropy) = features

        explanations.append((
            "URL Length", url_len,
            "High Risk (>= 54 chars)" if url_len >= 54 else "Normal (< 54 chars)",
            "Phishers use long URLs to hide deceptive parameters in the address bar." if url_len >= 54 else "Standard URL length."
        ))
        explanations.append((
            "Domain Length", domain_len,
            "Suspicious (>= 25 chars)" if domain_len >= 25 else "Normal",
            "Long domain names often stuff brand names and action words." if domain_len >= 25 else "Concise domain name."
        ))
        explanations.append((
            "IP Address in Host", "Yes" if has_ip else "No",
            "Critical Risk" if has_ip else "Safe",
            "Raw IP address used instead of domain name to evade DNS blacklists." if has_ip else "Standard domain hostname."
        ))
        explanations.append((
            "'@' Symbol Present", "Yes" if has_at else "No",
            "Critical Risk" if has_at else "Safe",
            "Browsers ignore everything before '@', redirecting to a hidden host." if has_at else "No '@' obfuscation detected."
        ))
        explanations.append((
            "Embedded '//' Redirect", "Yes" if double_slash else "No",
            "High Risk" if double_slash else "Safe",
            "Multiple '//' found in path, indicating open redirection to another site." if double_slash else "Clean URL path structure."
        ))
        explanations.append((
            "Hyphen (-) in Domain", "Yes" if has_hyphen else "No",
            "Moderate Risk" if has_hyphen else "Safe",
            "Hyphens in domains are frequently used for brand spoofing (e.g., pay-pal.com)." if has_hyphen else "No hyphenation in domain."
        ))
        explanations.append((
            "Subdomain Depth", subdomains,
            "High Risk (> 2)" if subdomains > 2 else "Safe",
            "Multiple nested subdomains attempt to mimic legitimate hierarchy." if subdomains > 2 else "Standard subdomain depth."
        ))
        explanations.append((
            "HTTPS Scheme", "Yes" if is_https else "No",
            "Safe" if is_https else "Moderate Risk",
            "Uses encrypted HTTPS protocol." if is_https else "Unencrypted HTTP or missing protocol scheme."
        ))
        explanations.append((
            "Fake 'https' in Domain", "Yes" if fake_https else "No",
            "Critical Risk" if fake_https else "Safe",
            "Deceptive 'https' text placed inside domain name to trick users." if fake_https else "No deceptive protocol tokens."
        ))
        explanations.append((
            "Total Dots (.)", dot_count,
            "Suspicious (> 3)" if dot_count > 3 else "Safe",
            "High dot count suggests deep subdomain chaining or file extensions." if dot_count > 3 else "Normal dot count."
        ))
        explanations.append((
            "Digit Ratio", f"{digit_ratio * 100:.1f}%",
            "High Risk (> 15%)" if digit_ratio > 0.15 else "Safe",
            "High proportion of numbers indicates random ID or IP spoofing." if digit_ratio > 0.15 else "Normal character composition."
        ))
        explanations.append((
            "Special Characters", special_chars,
            "Suspicious (> 4)" if special_chars > 4 else "Safe",
            "Heavy query parameters or obfuscated characters detected." if special_chars > 4 else "Minimal special characters."
        ))
        explanations.append((
            "Sensitive Keywords", kw_count,
            "High Risk (>= 2)" if kw_count >= 2 else ("Moderate (1)" if kw_count == 1 else "Safe"),
            f"Contains {kw_count} social-engineering keyword(s) (login, verify, secure, etc.)." if kw_count > 0 else "No urgent social-engineering words."
        ))
        explanations.append((
            "Suspicious TLD", "Yes" if susp_tld else "No",
            "High Risk" if susp_tld else "Safe",
            "Uses a high-abuse top-level domain (.tk, .xyz, .top, .ml, etc.)." if susp_tld else "Reputable top-level domain."
        ))
        explanations.append((
            "URL Shortener", "Yes" if is_shortener else "No",
            "Moderate Risk" if is_shortener else "Safe",
            "URL shortener masks the true destination domain." if is_shortener else "Direct destination URL."
        ))
        explanations.append((
            "Shannon Entropy", entropy,
            "High Risk (> 4.3)" if entropy > 4.3 else "Normal",
            "High randomness score typical of algorithmically generated phishing URLs." if entropy > 4.3 else "Predictable human-readable string."
        ))
        return explanations


# =====================================================================
# 2. SYNTHETIC DATASET GENERATOR & AI MODEL TRAINER
# =====================================================================
class PhishingDetectionAI:
    """Generates a balanced URL corpus and trains a Random Forest Classifier."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            random_state=42
        )
        self.metrics = {}
        self.feature_importances = {}
        self._build_and_train()

    def _generate_corpus(self):
        rng = random.Random(42)

        legit_domains = [
            "google.com", "github.com", "stackoverflow.com", "microsoft.com",
            "apple.com", "amazon.in", "wikipedia.org", "python.org", "linkedin.com",
            "netflix.com", "coursera.org", "kaggle.com", "openai.com", "bbc.co.uk",
            "nytimes.com", "HDFCbank.com", "sbi.co.in", "icicibank.com", "irctc.co.in",
            "flipkart.com", "zomato.com", "ncert.nic.in", "ugc.ac.in", "mit.edu",
            "stanford.edu", "ieee.org", "acm.org", "cloudflare.com", "mozilla.org",
            "ubuntu.com", "debian.org", "pytorch.org", "scikit-learn.org", "pandas.pydata.org"
        ]
        legit_paths = [
            "", "/about", "/docs/getting-started", "/products/cloud", "/news/2026-tech",
            "/research/publications", "/blog/data-science", "/courses/cyber-security",
            "/help/contact", "/careers/engineering", "/index.html", "/en/home",
            "/user/profile", "/repos/projects", "/articles/network-security"
        ]
        legit_subdomains = ["", "www.", "docs.", "blog.", "developer.", "research.", "news."]

        phish_brands = [
            "paypal", "appleid", "microsoft-365", "amazon-prime", "netflix-billing",
            "sbi-online", "hdfc-netbanking", "icici-verify", "google-drive",
            "metamask-wallet", "binance-auth", "chase-bank", "wellsfargo",
            "instagram-support", "facebook-security", "steam-community", "dhl-tracking"
        ]
        phish_words = [
            "secure", "login", "verify", "update", "account", "confirm",
            "unlock", "suspended", "kyc", "validation", "auth", "billing", "recover"
        ]
        phish_tlds = [".tk", ".xyz", ".top", ".ml", ".cf", ".gq", ".pw", ".buzz", ".work", ".click", ".com", ".net"]

        urls = []
        labels = []  # 0 = Legitimate, 1 = Phishing

        # Generate 220 Legitimate URLs
        for _ in range(220):
            sub = rng.choice(legit_subdomains)
            dom = rng.choice(legit_domains)
            path = rng.choice(legit_paths)
            if rng.random() < 0.25:
                path += f"?page={rng.randint(1, 9)}"
            scheme = "https://" if rng.random() < 0.94 else "http://"
            url = f"{scheme}{sub}{dom}{path}"
            urls.append(url)
            labels.append(0)

        # Generate 220 Diverse Phishing URLs across 5 realistic attack patterns
        for i in range(220):
            pattern = i % 5
            brand = rng.choice(phish_brands)
            w1, w2 = rng.sample(phish_words, 2)
            tld = rng.choice(phish_tlds)
            rand_hex = "".join(rng.choices("0123456789abcdef", k=14))

            if pattern == 0:
                # Hyphenated brand spoofing + suspicious TLD + sensitive path
                scheme = "http://" if rng.random() < 0.75 else "https://"
                url = f"{scheme}{brand}-{w1}-{w2}{tld}/auth/{w1}.php?session={rand_hex}&id={rng.randint(1000,9999)}"
            elif pattern == 1:
                # Raw IPv4 address hosting fake banking portal
                ip = f"{rng.randint(11, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
                url = f"http://{ip}/{brand}/{w1}-{w2}/index.html?token={rand_hex}"
            elif pattern == 2:
                # Deep subdomain nesting with fake 'https' token in hostname
                url = f"http://https-{brand}.{w1}.{w2}.service-portal{tld}/login/verify?user={rand_hex}"
            elif pattern == 3:
                # '@' credential obfuscation or embedded '//' open redirect
                if rng.random() < 0.5:
                    url = f"https://www.{brand}.com@{w1}-{w2}-server{tld}/signin? hash={rand_hex}"
                else:
                    url = f"http://redirect-portal{tld}/out//https://www.{brand}-{w1}.com/verify"
            else:
                # URL shortener or high-entropy obfuscated phishing link
                short_dom = rng.choice(["bit.ly", "tinyurl.com", "rb.gy", "t.ly"])
                if rng.random() < 0.45:
                    url = f"http://{short_dom}/{rand_hex[:8]}?redirect={brand}_{w1}"
                else:
                    url = f"https://auth-{w1}-{brand}-portal{tld}/?cmd=_signin&dispatch={rand_hex}{rand_hex}"

            urls.append(url)
            labels.append(1)

        return urls, labels

    def _build_and_train(self):
        urls, labels = self._generate_corpus()
        X = np.array([URLFeatureExtractor.extract_features(u) for u in urls])
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.22, stratify=y, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred) * 100,
            "precision": precision_score(y_test, y_pred) * 100,
            "recall": recall_score(y_test, y_pred) * 100,
            "f1": f1_score(y_test, y_pred) * 100,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        importances = self.model.feature_importances_
        for name, imp in zip(URLFeatureExtractor.FEATURE_NAMES, importances):
            self.feature_importances[name] = round(imp * 100, 2)

    def analyze_url(self, url: str) -> dict:
        features = URLFeatureExtractor.extract_features(url)
        X_vec = np.array([features])
        probabilities = self.model.predict_proba(X_vec)[0]
        phish_prob = float(probabilities[1]) * 100.0
        legit_prob = float(probabilities[0]) * 100.0

        if phish_prob >= 60.0:
            verdict = "PHISHING / MALICIOUS"
            risk_level = "CRITICAL RISK" if phish_prob >= 80.0 else "HIGH RISK"
            status_tag = "DANGER"
        elif phish_prob >= 38.0:
            verdict = "SUSPICIOUS URL"
            risk_level = "MODERATE RISK"
            status_tag = "WARNING"
        else:
            verdict = "LEGITIMATE / SAFE"
            risk_level = "LOW RISK"
            status_tag = "SAFE"

        explanations = URLFeatureExtractor.inspect_risk_factors(url, features)

        return {
            "url": url,
            "features": features,
            "phish_prob": round(phish_prob, 1),
            "legit_prob": round(legit_prob, 1),
            "verdict": verdict,
            "risk_level": risk_level,
            "status_tag": status_tag,
            "explanations": explanations,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }


# =====================================================================
# 3. MODERN DARK CYBERSECURITY GUI (TKINTER)
# =====================================================================
class PhishingDetectorGUI:
    """Cyber-themed Desktop GUI for Real-Time AI Phishing URL Analysis."""

    # Color Palette (Slate & Neon Cyber Theme)
    BG_MAIN = "#0B0F19"
    BG_CARD = "#151C2C"
    BG_INPUT = "#1E293B"
    FG_PRIMARY = "#F8FAFC"
    FG_MUTED = "#94A3B8"
    ACCENT_BLUE = "#38BDF8"
    ACCENT_INDIGO = "#6366F1"
    COLOR_SAFE = "#10B981"
    COLOR_WARN = "#F59E0B"
    COLOR_DANGER = "#EF4444"

    SAMPLE_URLS = [
        ("Safe: Official Python Docs", "https://docs.python.org/3/library/urllib.parse.html"),
        ("Safe: GitHub Repository", "https://github.com/scikit-learn/scikit-learn"),
        ("Safe: SBI Corporate Portal", "https://www.sbi.co.in/web/personal-banking"),
        ("Phishing: Spoofed PayPal Login", "http://paypal-secure-account-verify.tk/auth/login.php?session=9a8f7b6c5d4e3f"),
        ("Phishing: Raw IP Banking Trap", "http://192.168.45.211/sbi-online/kyc-update/index.html?token=88412a9b"),
        ("Phishing: '@' Credential Redirection", "https://www.google.com@secure-login-update-server.xyz/signin?id=49102"),
        ("Phishing: Fake HTTPS Subdomain", "http://https-microsoft-365.account.verify.service-portal.top/login/auth")
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AegisURL // AI Phishing & Malicious Website Detector")
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)
        self.root.configure(bg=self.BG_MAIN)

        # Initialize AI Engine
        self.ai_engine = PhishingDetectionAI()
        self.scan_history = []

        self._configure_styles()
        self._build_header()
        self._build_input_bar()
        self._build_main_workspace()
        self._build_footer()

        # Run an initial demo scan on startup
        self.url_var.set(self.SAMPLE_URLS[0][1])
        self.run_scan()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.BG_MAIN)
        style.configure("Card.TFrame", background=self.BG_CARD)

        # Notebook (Tabs) styling
        style.configure(
            "Cyber.TNotebook",
            background=self.BG_MAIN,
            borderwidth=0
        )
        style.configure(
            "Cyber.TNotebook.Tab",
            background=self.BG_INPUT,
            foreground=self.FG_MUTED,
            padding=[16, 8],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map(
            "Cyber.TNotebook.Tab",
            background=[("selected", self.ACCENT_INDIGO)],
            foreground=[("selected", "#FFFFFF")]
        )

        # Treeview styling
        style.configure(
            "Cyber.Treeview",
            background=self.BG_CARD,
            foreground=self.FG_PRIMARY,
            fieldbackground=self.BG_CARD,
            rowheight=29,
            font=("Segoe UI", 9),
            borderwidth=0
        )
        style.configure(
            "Cyber.Treeview.Heading",
            background=self.BG_INPUT,
            foreground=self.ACCENT_BLUE,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padding=6
        )
        style.map(
            "Cyber.Treeview",
            background=[("selected", "#2563EB")],
            foreground=[("selected", "#FFFFFF")]
        )

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=self.BG_MAIN, padx=24, pady=14)
        header_frame.pack(fill="x")

        title_box = tk.Frame(header_frame, bg=self.BG_MAIN)
        title_box.pack(side="left")

        tk.Label(
            title_box,
            text="🛡️  AI PHISHING WEBSITE DETECTOR",
            font=("Segoe UI", 16, "bold"),
            bg=self.BG_MAIN,
            fg=self.ACCENT_BLUE
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="Real-Time Lexical Feature Extraction & Random Forest Threat Classification Engine",
            font=("Segoe UI", 9),
            bg=self.BG_MAIN,
            fg=self.FG_MUTED
        ).pack(anchor="w")

        # Model Accuracy Badge on Right
        badge_frame = tk.Frame(header_frame, bg=self.BG_CARD, padx=14, pady=8, highlightthickness=1, highlightbackground=self.ACCENT_INDIGO)
        badge_frame.pack(side="right")

        acc = self.ai_engine.metrics["accuracy"]
        f1 = self.ai_engine.metrics["f1"]
        tk.Label(
            badge_frame,
            text=f"MODEL READY  •  ACC: {acc:.1f}%  •  F1: {f1:.1f}%",
            font=("Consolas", 10, "bold"),
            bg=self.BG_CARD,
            fg=self.COLOR_SAFE
        ).pack()

    def _build_input_bar(self):
        bar_card = tk.Frame(
            self.root, bg=self.BG_CARD, padx=20, pady=16,
            highlightthickness=1, highlightbackground="#1E293B"
        )
        bar_card.pack(fill="x", padx=24, pady=(0, 14))

        # Top row: URL Entry + Buttons
        top_row = tk.Frame(bar_card, bg=self.BG_CARD)
        top_row.pack(fill="x")

        tk.Label(
            top_row, text="TARGET URL:", font=("Segoe UI", 10, "bold"),
            bg=self.BG_CARD, fg=self.FG_PRIMARY
        ).pack(side="left", padx=(0, 10))

        self.url_var = tk.StringVar()
        self.url_entry = tk.Entry(
            top_row,
            textvariable=self.url_var,
            font=("Consolas", 11),
            bg=self.BG_INPUT,
            fg=self.FG_PRIMARY,
            insertbackground=self.ACCENT_BLUE,
            relief="flat",
            bd=8
        )
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.url_entry.bind("<Return>", lambda event: self.run_scan())

        scan_btn = tk.Button(
            top_row,
            text="⚡ ANALYZE URL",
            command=self.run_scan,
            font=("Segoe UI", 10, "bold"),
            bg=self.ACCENT_INDIGO,
            fg="#FFFFFF",
            activebackground=self.ACCENT_BLUE,
            activeforeground="#0B0F19",
            relief="flat",
            padx=18,
            pady=6,
            cursor="hand2"
        )
        scan_btn.pack(side="left", padx=(0, 8))

        clear_btn = tk.Button(
            top_row,
            text="CLEAR",
            command=self._clear_input,
            font=("Segoe UI", 9, "bold"),
            bg=self.BG_INPUT,
            fg=self.FG_MUTED,
            activebackground="#334155",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2"
        )
        clear_btn.pack(side="left")

        # Bottom row: Quick Sample Selector
        sample_row = tk.Frame(bar_card, bg=self.BG_CARD)
        sample_row.pack(fill="x", pady=(10, 0))

        tk.Label(
            sample_row, text="QUICK TEST PRESETS:", font=("Segoe UI", 8, "bold"),
            bg=self.BG_CARD, fg=self.FG_MUTED
        ).pack(side="left", padx=(0, 8))

        self.sample_combo = ttk.Combobox(
            sample_row,
            values=[f"{label}  -->  {u}" for label, u in self.SAMPLE_URLS],
            state="readonly",
            font=("Segoe UI", 9),
            width=75
        )
        self.sample_combo.current(0)
        self.sample_combo.pack(side="left", padx=(0, 8))
        self.sample_combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

    def _build_main_workspace(self):
        workspace = tk.Frame(self.root, bg=self.BG_MAIN)
        workspace.pack(fill="both", expand=True, padx=24, pady=(0, 10))

        # Left Panel: Risk Gauge & Verdict Summary
        left_card = tk.Frame(
            workspace, bg=self.BG_CARD, width=330, padx=18, pady=18,
            highlightthickness=1, highlightbackground="#1E293B"
        )
        left_card.pack(side="left", fill="y", padx=(0, 14))
        left_card.pack_propagate(False)

        tk.Label(
            left_card, text="THREAT SCORE & VERDICT",
            font=("Segoe UI", 10, "bold"), bg=self.BG_CARD, fg=self.FG_MUTED
        ).pack(anchor="w")

        # Canvas for Semi-Circular Threat Meter
        self.gauge_canvas = tk.Canvas(
            left_card, width=280, height=175, bg=self.BG_CARD, highlightthickness=0
        )
        self.gauge_canvas.pack(pady=(10, 6))

        self.verdict_label = tk.Label(
            left_card, text="READY", font=("Segoe UI", 14, "bold"),
            bg=self.BG_CARD, fg=self.COLOR_SAFE
        )
        self.verdict_label.pack(pady=(2, 2))

        self.risk_badge = tk.Label(
            left_card, text="LOW RISK", font=("Consolas", 10, "bold"),
            bg=self.BG_INPUT, fg=self.FG_PRIMARY, padx=10, pady=3
        )
        self.risk_badge.pack(pady=(0, 14))

        # Divider
        tk.Frame(left_card, bg="#1E293B", height=1).pack(fill="x", pady=8)

        tk.Label(
            left_card, text="KEY DIAGNOSTIC SUMMARY",
            font=("Segoe UI", 9, "bold"), bg=self.BG_CARD, fg=self.ACCENT_BLUE
        ).pack(anchor="w", pady=(4, 6))

        self.summary_text = tk.Label(
            left_card,
            text="Enter any URL above to inspect its lexical structure and AI threat probability.",
            font=("Segoe UI", 9),
            bg=self.BG_CARD,
            fg=self.FG_PRIMARY,
            justify="left",
            wraplength=285
        )
        self.summary_text.pack(anchor="w", fill="x")

        # Right Panel: Tabbed Detailed Inspection
        right_panel = tk.Frame(workspace, bg=self.BG_MAIN)
        right_panel.pack(side="left", fill="both", expand=True)

        self.notebook = ttk.Notebook(right_panel, style="Cyber.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: 16-Feature Lexical Breakdown
        tab_features = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.notebook.add(tab_features, text=" 🔍 Lexical Feature Breakdown (16) ")
        self._build_features_tab(tab_features)

        # Tab 2: Scan History Log
        tab_history = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.notebook.add(tab_history, text=" 📜 Live Scan History ")
        self._build_history_tab(tab_history)

        # Tab 3: AI Model & Feature Importance
        tab_model = tk.Frame(self.notebook, bg=self.BG_CARD)
        self.notebook.add(tab_model, text=" 🧠 Random Forest AI Metrics ")
        self._build_model_tab(tab_model)

    def _build_features_tab(self, parent):
        cols = ("Feature", "Value", "Assessment", "Security Insight")
        self.feature_tree = ttk.Treeview(
            parent, columns=cols, show="headings", style="Cyber.Treeview"
        )
        self.feature_tree.heading("Feature", text="Lexical Indicator")
        self.feature_tree.heading("Value", text="Extracted Value")
        self.feature_tree.heading("Assessment", text="Status")
        self.feature_tree.heading("Security Insight", text="Cybersecurity Interpretation")

        self.feature_tree.column("Feature", width=155, anchor="w")
        self.feature_tree.column("Value", width=105, anchor="center")
        self.feature_tree.column("Assessment", width=140, anchor="center")
        self.feature_tree.column("Security Insight", width=340, anchor="w")

        # Row coloring tags
        self.feature_tree.tag_configure("crit", foreground="#FCA5A5", background="#2A1215")
        self.feature_tree.tag_configure("warn", foreground="#FCD34D", background="#292112")
        self.feature_tree.tag_configure("safe", foreground="#A7F3D0", background=self.BG_CARD)

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.feature_tree.yview)
        self.feature_tree.configure(yscrollcommand=scrollbar.set)

        self.feature_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", pady=8, padx=(0, 8))

    def _build_history_tab(self, parent):
        cols = ("Time", "Verdict", "Threat %", "URL")
        self.history_tree = ttk.Treeview(
            parent, columns=cols, show="headings", style="Cyber.Treeview"
        )
        self.history_tree.heading("Time", text="Timestamp")
        self.history_tree.heading("Verdict", text="AI Verdict")
        self.history_tree.heading("Threat %", text="Phishing Prob.")
        self.history_tree.heading("URL", text="Analyzed URL")

        self.history_tree.column("Time", width=90, anchor="center")
        self.history_tree.column("Verdict", width=165, anchor="center")
        self.history_tree.column("Threat %", width=110, anchor="center")
        self.history_tree.column("URL", width=390, anchor="w")

        self.history_tree.tag_configure("DANGER", foreground=self.COLOR_DANGER)
        self.history_tree.tag_configure("WARNING", foreground=self.COLOR_WARN)
        self.history_tree.tag_configure("SAFE", foreground=self.COLOR_SAFE)

        self.history_tree.bind("<Double-1>", self._reload_from_history)

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)

        self.history_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", pady=8, padx=(0, 8))

    def _build_model_tab(self, parent):
        container = tk.Frame(parent, bg=self.BG_CARD, padx=18, pady=14)
        container.pack(fill="both", expand=True)

        m = self.ai_engine.metrics
        stats_str = (
            f"Classifier: Scikit-Learn RandomForestClassifier (120 Trees, Max Depth=12)   |   "
            f"Train Samples: {m['train_samples']} URLs   |   Test Samples: {m['test_samples']} URLs\n"
            f"Validation Accuracy: {m['accuracy']:.2f}%   |   Precision: {m['precision']:.2f}%   |   "
            f"Recall: {m['recall']:.2f}%   |   F1-Score: {m['f1']:.2f}%"
        )
        tk.Label(
            container, text=stats_str, font=("Consolas", 9, "bold"),
            bg=self.BG_INPUT, fg=self.ACCENT_BLUE, justify="left", padx=12, pady=10
        ).pack(fill="x", pady=(0, 12))

        tk.Label(
            container,
            text="RANDOM FOREST FEATURE IMPORTANCE RANKING (GINI IMPURITY REDUCTION):",
            font=("Segoe UI", 9, "bold"), bg=self.BG_CARD, fg=self.FG_MUTED
        ).pack(anchor="w", pady=(0, 6))

        imp_canvas = tk.Canvas(container, bg=self.BG_CARD, highlightthickness=0)
        imp_canvas.pack(fill="both", expand=True)

        sorted_imp = sorted(
            self.ai_engine.feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:10]

        max_val = max(val for _, val in sorted_imp) if sorted_imp else 1.0
        for idx, (fname, val) in enumerate(sorted_imp):
            y_pos = 12 + idx * 28
            imp_canvas.create_text(
                5, y_pos + 8, text=f"{idx+1}. {fname}", anchor="w",
                fill=self.FG_PRIMARY, font=("Segoe UI", 9)
            )
            bar_max_width = 380
            bar_w = int((val / max_val) * bar_max_width)
            imp_canvas.create_rectangle(
                185, y_pos, 185 + bar_w, y_pos + 16,
                fill=self.ACCENT_INDIGO, outline=""
            )
            imp_canvas.create_text(
                195 + bar_w, y_pos + 8, text=f"{val:.1f}%", anchor="w",
                fill=self.ACCENT_BLUE, font=("Consolas", 9, "bold")
            )

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=self.BG_MAIN, padx=24, pady=6)
        footer.pack(fill="x", side="bottom")
        tk.Label(
            footer,
            text="Tip: Double-click any entry in 'Live Scan History' to reload its full feature report.",
            font=("Segoe UI", 8), bg=self.BG_MAIN, fg=self.FG_MUTED
        ).pack(side="left")

    def _draw_gauge(self, phish_prob: float, status_tag: str):
        self.gauge_canvas.delete("all")
        color = (
            self.COLOR_DANGER if status_tag == "DANGER"
            else (self.COLOR_WARN if status_tag == "WARNING" else self.COLOR_SAFE)
        )

        # Background Arc
        self.gauge_canvas.create_arc(
            30, 20, 250, 240, start=0, extent=180,
            style="arc", outline=self.BG_INPUT, width=22
        )

        # Active Threat Arc (drawn counter-clockwise from 180 degrees)
        extent_angle = -int((phish_prob / 100.0) * 180)
        if abs(extent_angle) > 0:
            self.gauge_canvas.create_arc(
                30, 20, 250, 240, start=180, extent=extent_angle,
                style="arc", outline=color, width=22
            )

        # Center Percentage Readout
        self.gauge_canvas.create_text(
            140, 105, text=f"{phish_prob:.1f}%",
            font=("Consolas", 26, "bold"), fill=color
        )
        self.gauge_canvas.create_text(
            140, 138, text="PHISHING PROBABILITY",
            font=("Segoe UI", 8, "bold"), fill=self.FG_MUTED
        )

        # Scale Labels
        self.gauge_canvas.create_text(30, 155, text="0% (Safe)", font=("Segoe UI", 8), fill=self.FG_MUTED)
        self.gauge_canvas.create_text(245, 155, text="100% (Phish)", font=("Segoe UI", 8), fill=self.FG_MUTED)

    def _on_sample_selected(self, event=None):
        idx = self.sample_combo.current()
        if 0 <= idx < len(self.SAMPLE_URLS):
            self.url_var.set(self.SAMPLE_URLS[idx][1])
            self.run_scan()

    def _clear_input(self):
        self.url_var.set("")
        self.url_entry.focus_set()

    def _reload_from_history(self, event=None):
        selection = self.history_tree.selection()
        if not selection:
            return
        item_vals = self.history_tree.item(selection[0], "values")
        if len(item_vals) >= 4:
            self.url_var.set(item_vals[3])
            self.run_scan()
            self.notebook.select(0)

    def run_scan(self):
        raw_url = self.url_var.get().strip()
        if not raw_url:
            messagebox.showwarning("Input Required", "Please enter a URL to analyze.")
            return

        result = self.ai_engine.analyze_url(raw_url)
        tag = result["status_tag"]
        color = (
            self.COLOR_DANGER if tag == "DANGER"
            else (self.COLOR_WARN if tag == "WARNING" else self.COLOR_SAFE)
        )

        # Update Gauge and Labels
        self._draw_gauge(result["phish_prob"], tag)
        self.verdict_label.configure(text=result["verdict"], fg=color)
        self.risk_badge.configure(
            text=f"{result['risk_level']}  •  SAFE SCORE: {result['legit_prob']:.1f}%",
            fg=color
        )

        # Build Executive Diagnostic Summary
        triggered = [
            f"• {feat}: {insight}"
            for feat, val, status, insight in result["explanations"]
            if any(k in status for k in ["Critical", "High", "Suspicious", "Moderate"])
        ]
        if triggered:
            summary_str = "Detected Risk Indicators:\n" + "\n".join(triggered[:4])
            if len(triggered) > 4:
                summary_str += f"\n...and {len(triggered) - 4} more indicator(s) in the breakdown tab."
        else:
            summary_str = (
                "✓ Clean lexical structure.\n"
                "✓ Standard domain length & character entropy.\n"
                "✓ No deceptive tokens, raw IPs, or suspicious TLDs detected."
            )
        self.summary_text.configure(text=summary_str)

        # Populate 16-Feature Treeview
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        for feat, val, status, insight in result["explanations"]:
            if "Critical" in status or "High" in status:
                row_tag = "crit"
            elif "Moderate" in status or "Suspicious" in status:
                row_tag = "warn"
            else:
                row_tag = "safe"
            self.feature_tree.insert("", "end", values=(feat, val, status, insight), tags=(row_tag,))

        # Prepend to Scan History
        self.history_tree.insert(
            "", 0,
            values=(result["timestamp"], result["verdict"], f"{result['phish_prob']:.1f}%", raw_url),
            tags=(tag,)
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = PhishingDetectorGUI(root)
    root.mainloop()


