"""Local (offline) transaction categorization and flow classification.

Pluggy categorization is a paid feature: after the 15-day trial ``category`` arrives as ``null``.
To keep monthly aggregates consistent before and after the trial, every transaction receives a
*local* Brazilian-Portuguese category computed here:

1. Pluggy's English category (when present) mapped to the local pt-BR taxonomy;
2. otherwise — or when Pluggy's category is too generic — keyword rules on the description;
3. otherwise ``"Outros"``.

All functions are pure and thread-safe.
"""

from __future__ import annotations

import re
import unicodedata

OTHER = "Outros"
CARD_PAYMENT = "Pagamento de fatura"
OWN_TRANSFER = "Transferência entre contas próprias"
TRANSFERS = "Transferências"
INVESTMENTS = "Investimentos"

#: Pluggy (English) category description -> local pt-BR category.
PLUGGY_TO_LOCAL: dict[str, str] = {
    "income": "Renda",
    "salary": "Salário",
    "retirement": "Aposentadoria",
    "entrepreneurial activities": "Renda",
    "government aid": "Auxílio governamental",
    "non-recurring income": "Renda",
    "loans and financing": "Empréstimos e financiamentos",
    "loans": "Empréstimos e financiamentos",
    "financing": "Empréstimos e financiamentos",
    "late payment and overdraft costs": "Juros e encargos",
    "interests charged": "Juros e encargos",
    "investments": INVESTMENTS,
    "automatic investment": INVESTMENTS,
    "fixed income": INVESTMENTS,
    "mutual funds": INVESTMENTS,
    "variable income": INVESTMENTS,
    "margin": INVESTMENTS,
    "pension": INVESTMENTS,
    "proceeds interests and dividends": "Rendimentos",
    "same person transfer": OWN_TRANSFER,
    "transfers": TRANSFERS,
    "transfer": TRANSFERS,
    "credit card payment": CARD_PAYMENT,
    "services": "Serviços",
    "telecommunications": "Telecomunicações",
    "internet": "Telecomunicações",
    "mobile": "Telecomunicações",
    "tv": "Telecomunicações",
    "education": "Educação",
    "online courses": "Educação",
    "wellness and fitness": "Academia e bem-estar",
    "wellness": "Academia e bem-estar",
    "gyms and fitness centers": "Academia e bem-estar",
    "tickets": "Lazer",
    "shopping": "Compras",
    "online shopping": "Compras online",
    "electronics": "Eletrônicos",
    "pet supplies and vet": "Pets",
    "clothing": "Vestuário",
    "kids and toys": "Compras",
    "bookstore": "Compras",
    "sports goods": "Compras",
    "office supplies": "Compras",
    "cashback": "Cashback",
    "digital services": "Serviços digitais",
    "gaming": "Jogos",
    "video streaming": "Streaming de vídeo",
    "music streaming": "Streaming de música",
    "food and drinks": "Alimentação",
    "eating out": "Restaurantes",
    "food delivery": "Delivery de comida",
    "groceries": "Supermercado",
    "taxes": "Impostos e taxas",
    "income taxes": "Impostos e taxas",
    "taxes on investments": "Impostos e taxas",
    "tax on financial operations": "Impostos e taxas",
    "bank fees": "Tarifas bancárias",
    "housing": "Moradia",
    "utilities": "Contas de consumo",
    "rent": "Moradia",
    "healthcare": "Saúde",
    "pharmacy": "Saúde",
    "transportation": "Transporte",
    "taxi and ride-hailing": "Transporte",
    "automotive": "Transporte",
    "gas stations": "Transporte",
    "parking": "Transporte",
    "insurance": "Seguros",
    "leisure": "Lazer",
    "travel": "Viagem",
    "donations": "Doações",
    "gambling": "Apostas",
    "legal obligations": "Obrigações legais",
}

#: Pluggy categories too broad to trust over a specific keyword match.
GENERIC_LOCAL = frozenset({"Compras", "Serviços", "Alimentação", TRANSFERS, OTHER, "Renda"})

#: Local categories whose charges are typically subscriptions / recurring services.
SUBSCRIPTION_CATEGORIES = frozenset(
    {
        "Streaming de vídeo",
        "Streaming de música",
        "Serviços digitais",
        "Jogos",
        "Telecomunicações",
        "Academia e bem-estar",
        "Educação",
        "Seguros",
    }
)

# Ordered: first match wins. Patterns run on the accent-stripped, lower-cased description.
_KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    (r"pagamento (de )?fatura|pagto fatura|pag fatura|pgto fatura|fatura cartao|pagamento recebido", CARD_PAYMENT),
    (r"mesma titularidade|entre contas|transf.* propria", OWN_TRANSFER),
    (r"amazon ?prime|prime ?video|primevideo", "Streaming de vídeo"),
    (
        r"netflix|disney|hbo|max\.com|globoplay|paramount|star\+|crunchyroll|apple ?tv|youtube ?premium|mubi",
        "Streaming de vídeo",
    ),
    (r"spotify|deezer|apple ?music|tidal|youtube ?music", "Streaming de música"),
    (
        (
            r"icloud|apple\.com|google ?one|google ?storage|microsoft|office ?365|adobe|dropbox|chatgpt|openai|"
            r"anthropic|claude\.ai|github|notion|canva|linkedin"
        ),
        "Serviços digitais",
    ),
    (r"steam|playstation|psn|xbox|nintendo|epic ?games", "Jogos"),
    (r"ifood|rappi|uber ?eats|aiqfome|ze delivery|james delivery", "Delivery de comida"),
    (
        (
            r"amazon|amzn|mercado ?livre|mercadolivre|shopee|aliexpress|shein|magalu|magazine ?luiza|americanas|"
            r"submarino|casas ?bahia|kabum|temu|netshoes|dafiti|centauro"
        ),
        "Compras online",
    ),
    (
        r"uber|99 ?app|99 ?pop|99 ?taxi|cabify|posto|shell|ipiranga|petrobras|estapar|sem ?parar|conectcar|veloe",
        "Transporte",
    ),
    (r"carrefour|pao de acucar|assai|atacadao|supermercado|hortifruti|sams club|mercado extra", "Supermercado"),
    (r"restaurante|lanchonete|padaria|mcdonald|burger ?king|starbucks|outback|habib|subway", "Restaurantes"),
    (r"\bvivo\b|\bclaro\b|\btim\b|\boi\b|sky ?brasil|net servicos", "Telecomunicações"),
    (r"enel|light s|cemig|copel|celesc|sabesp|cedae|comgas|copasa|energia|saneamento", "Contas de consumo"),
    (r"condominio|aluguel|imobiliaria|quinto ?andar", "Moradia"),
    (
        (
            r"farmacia|drogaria|droga ?raia|drogasil|pague ?menos|unimed|amil|sulamerica|hapvida|hospital|"
            r"laboratorio|clinica"
        ),
        "Saúde",
    ),
    (r"smart ?fit|bluefit|gympass|wellhub|totalpass|academia", "Academia e bem-estar"),
    (r"escola|faculdade|universidade|udemy|alura|coursera|duolingo|curso", "Educação"),
    (r"seguro|porto seguro|seguradora", "Seguros"),
    (r"\biof\b|imposto|darf|ipva|iptu", "Impostos e taxas"),
    (r"tarifa|anuidade|cesta de servicos", "Tarifas bancárias"),
    (r"juros(?! sobre capital)|encargo|\bmora\b|multa", "Juros e encargos"),
    (r"latam|gol linhas|azul linhas|airbnb|booking|decolar|hotel|123 ?milhas", "Viagem"),
    (r"salario|folha de pagamento|proventos|pagamento de salario", "Salário"),
    (r"rendimento|dividendo|juros sobre capital", "Rendimentos"),
    (r"aplicacao|resgate|\bcdb\b|tesouro|\blci\b|\blca\b|corretora|poupanca", INVESTMENTS),
    (r"\bpix\b|\bted\b|\bdoc\b|transferencia", TRANSFERS),
)
_COMPILED_RULES = tuple((re.compile(pattern), category) for pattern, category in _KEYWORD_RULES)

_INSTALLMENT_RE = re.compile(r"\b(parc(ela)?\.?\s*)?\d{1,2}\s*/\s*\d{1,2}\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9+]+")
_DIGITS_RE = re.compile(r"\d+")
_NOISE_PREFIXES = ("compra no debito ", "compra cartao ", "pag ", "pg ", "pagamento ", "pix enviado ", "pix ")


def strip_accents(text: str) -> str:
    """Remove diacritics and lower-case a string.

    Args:
        text: Input text.

    Returns:
        str: ASCII-folded lower-case text.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def merchant_key(description: str | None, merchant: str | None = None) -> str:
    """Build a stable grouping key for a counterparty (used to detect recurring charges).

    Args:
        description: Transaction description as printed by the bank.
        merchant: Merchant name from Pluggy enrichment, when available.

    Returns:
        str: Normalized key such as ``"netflix com"`` (empty string when nothing usable).
    """
    base = strip_accents(merchant or description or "")
    base = _INSTALLMENT_RE.sub(" ", base)
    base = _DIGITS_RE.sub(" ", base)
    base = _NON_ALNUM_RE.sub(" ", base).strip()
    for prefix in _NOISE_PREFIXES:
        if base.startswith(prefix) and len(base) > len(prefix) + 2:
            base = base[len(prefix) :]
    return " ".join(base.split()[:4])


def map_pluggy_category(category: str | None) -> str | None:
    """Map a Pluggy category description (e.g. ``"Transfer - PIX"``) to the local taxonomy.

    Args:
        category: Pluggy ``category`` field (English).

    Returns:
        str | None: Local pt-BR category, or ``None`` when unknown/empty.
    """
    if not category:
        return None
    key = category.strip().lower()
    if key in PLUGGY_TO_LOCAL:
        return PLUGGY_TO_LOCAL[key]
    head = key.split(" - ", 1)[0].strip()
    if head in PLUGGY_TO_LOCAL:
        return PLUGGY_TO_LOCAL[head]
    if head.startswith("transfer"):
        return TRANSFERS
    return None


def keyword_category(description: str | None) -> str | None:
    """Categorize a transaction from its description using keyword rules.

    Args:
        description: Transaction description.

    Returns:
        str | None: Local category, or ``None`` when no rule matches.
    """
    if not description:
        return None
    text = strip_accents(description)
    for pattern, category in _COMPILED_RULES:
        if pattern.search(text):
            return category
    return None


def local_category(category: str | None, description: str | None, translated: str | None = None) -> str:
    """Compute the local pt-BR category of a transaction.

    Args:
        category: Pluggy category (English), may be ``None``.
        description: Transaction description.
        translated: Pluggy's ``descriptionTranslated`` for the category, if known.

    Returns:
        str: Local category (never empty).
    """
    mapped = map_pluggy_category(category)
    if mapped is None and category:
        mapped = translated or category
    if mapped is None or mapped in GENERIC_LOCAL:
        by_keyword = keyword_category(description)
        if by_keyword is not None and (mapped is None or by_keyword not in {TRANSFERS}):
            return by_keyword
    return mapped or OTHER


def classify_kind(flow: float, category_local: str, account_type: str | None) -> str:
    """Classify a transaction's economic nature.

    Args:
        flow: Signed amount (negative = money out of the account / card charge).
        category_local: Local category from :func:`local_category`.
        account_type: Pluggy account type (``BANK`` or ``CREDIT``).

    Returns:
        str: One of ``expense``, ``refund``, ``income``, ``transfer`` (card bill payments and
        own-account transfers, excluded from spending to avoid double counting) or ``investment``.
    """
    if category_local in {CARD_PAYMENT, OWN_TRANSFER}:
        return "transfer"
    if category_local == INVESTMENTS:
        return "investment"
    if flow < 0:
        return "expense"
    if (account_type or "").upper() == "CREDIT":
        return "refund"
    return "income"


def signed_flow(amount: float | None, tx_type: str | None, account_type: str | None) -> float:
    """Return the amount signed from the account holder's perspective.

    Pluggy normalizes ``type`` so purchases/outflows are always ``DEBIT`` (also on credit cards,
    whose raw amounts are positive for charges). ``type`` is therefore the source of truth.

    Args:
        amount: Raw Pluggy ``amount``.
        tx_type: ``DEBIT`` or ``CREDIT``.
        account_type: ``BANK`` or ``CREDIT``.

    Returns:
        float: Negative for money out (spending), positive for money in.
    """
    value = abs(float(amount or 0.0))
    kind = (tx_type or "").upper()
    if kind == "DEBIT":
        return -value
    if kind == "CREDIT":
        return value
    raw = float(amount or 0.0)
    return -raw if (account_type or "").upper() == "CREDIT" else raw
