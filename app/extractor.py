import re
from dateutil.parser import parse

def extract_fields(text):
    fields = {}

    # Name: allow middle initials
    name_match = re.findall(r"[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+", text)
    fields['name'] = name_match[0] if name_match else ""

    # Date: multiple formats
    date_match = re.findall(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}", text)
    fields['date'] = date_match[0] if date_match else ""

    # Amount: ₹, $, commas, decimals
    amount_match = re.findall(r"(₹|\$)?\d[\d,]*\.?\d*", text)
    fields['amount'] = amount_match[0] if amount_match else ""

    # ID Number
    id_match = re.findall(r"[A-Z0-9]{5,15}", text)
    fields['id_number'] = id_match[0] if id_match else ""

    # Phone number: with or without country code
    phone_match = re.findall(r"\+?\d{10,14}", text)
    fields['phone'] = phone_match[0] if phone_match else ""

    # Email
    email_match = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    fields['email'] = email_match[0] if email_match else ""

    return fields

# def extract_fields(text):
#     fields = {}

#     # ----------------------------
#     # Name (detect proper names)
#     # ----------------------------
#     # Match typical names: First Last (capitalized)
#     name_match = re.findall(r"\b[A-Z][a-z]{1,20} [A-Z][a-z]{1,20}\b", text)
#     fields['name'] = name_match[0] if name_match else ""

#     # ----------------------------
#     # Date (multiple formats)
#     # ----------------------------
#     date_match = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{1,2} [A-Za-z]{3,9} \d{4}\b", text)
#     # Use dateutil to normalize
#     fields['date'] = ""
#     if date_match:
#         try:
#             fields['date'] = str(parse(date_match[0], dayfirst=True).date())
#         except:
#             fields['date'] = date_match[0]

#     # ----------------------------
#     # Amount (currency)
#     # ----------------------------
#     amount_match = re.findall(r"₹\s?\d[\d,]*\.?\d*|\$\s?\d[\d,]*\.?\d*|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
#     fields['amount'] = amount_match[0] if amount_match else ""

#     # ----------------------------
#     # ID Number (alphanumeric, 5-15 chars)
#     # ----------------------------
#     id_match = re.findall(r"\b[A-Z0-9]{5,15}\b", text)
#     fields['id_number'] = id_match[0] if id_match else ""

#     # ----------------------------
#     # Phone number (optional)
#     # ----------------------------
#     phone_match = re.findall(r"\+?\d{1,3}[-\s]?\d{10}\b|\b\d{10}\b", text)
#     fields['phone'] = phone_match[0] if phone_match else ""

#     # ----------------------------
#     # Email (optional)
#     # ----------------------------
#     email_match = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
#     fields['email'] = email_match[0] if email_match else ""

#     return fields
