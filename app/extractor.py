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


