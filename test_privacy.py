# test_privacy.py
from privacy_layer import mask_text

test = """
Contact: ananya.sharma@infocorp.com - +91 98201 34567
Aadhaar: 4521 8834 9901  PAN: ABCPN1234F
ESG CapEx: USD 45.6 million
Projected ROI: 18.5% IRR over 10 years
Budget: USD 8.5 million (23.4% CAGR)
IP: 192.168.1.105
"""

result = mask_text(test, role="guest")
print("Masked:\n", result.masked_text)
print("\nStats:", result.stats)
print("\nLog:", result.redaction_log)