import json
from pathlib import Path

RAW_TICKETS = [
    # ==================== SHIPPING (12: 4 High, 4 Medium, 4 Low) ====================
    {"id": "TKT-001", "ticket_text": "My package for order ORD-1001 was marked as delivered two hours ago, but nothing arrived at my doorstep and it contains urgent medication.", "category": "Shipping", "order_id": "ORD-1001", "priority": "High"},
    {"id": "TKT-002", "ticket_text": "Order ORD-1002 is stuck in customs clearance for 10 days and I need these presentation materials before tomorrow morning's conference.", "category": "Shipping", "order_id": "ORD-1002", "priority": "High"},
    {"id": "TKT-003", "ticket_text": "The courier delivered order ORD-1003 to the completely wrong address across town. Please intervene immediately before it gets lost.", "category": "Shipping", "order_id": "ORD-1003", "priority": "High"},
    {"id": "TKT-004", "ticket_text": "Urgent: the tracking link for order ORD-1004 now says returned to sender due to a damaged label, and this is a birthday gift for tonight.", "category": "Shipping", "order_id": "ORD-1004", "priority": "High"},
    {"id": "TKT-005", "ticket_text": "Can you check why the tracking status for order ORD-1005 has not updated since leaving the sorting facility three days ago?", "category": "Shipping", "order_id": "ORD-1005", "priority": "Medium"},
    {"id": "TKT-006", "ticket_text": "I noticed a typo in my street number for order ORD-1006. Could you update the delivery address before it ships out?", "category": "Shipping", "order_id": "ORD-1006", "priority": "Medium"},
    {"id": "TKT-007", "ticket_text": "Part of my items from order ORD-1007 arrived today, but the second box is still missing from the shipment.", "category": "Shipping", "order_id": "ORD-1007", "priority": "Medium"},
    {"id": "TKT-008", "ticket_text": "My order ORD-1008 was supposed to arrive yesterday via standard delivery, can you provide a new estimated delivery date?", "category": "Shipping", "order_id": "ORD-1008", "priority": "Medium"},
    {"id": "TKT-009", "ticket_text": "Do you offer international shipping to Canada and Tunisia for standard electronics orders?", "category": "Shipping", "order_id": None, "priority": "Low"},
    {"id": "TKT-010", "ticket_text": "Where can I find the carrier tracking number for my recent order ORD-1010? No rush, just want to follow it.", "category": "Shipping", "order_id": "ORD-1010", "priority": "Low"},
    {"id": "TKT-011", "ticket_text": "What are your usual shipping fees and delivery timeframes for rural areas?", "category": "Shipping", "order_id": None, "priority": "Low"},
    {"id": "TKT-012", "ticket_text": "Could you confirm whether order ORD-1012 will require a signature upon delivery next week?", "category": "Shipping", "order_id": "ORD-1012", "priority": "Low"},

    # ==================== REFUND (12: 4 High, 4 Medium, 4 Low) ====================
    {"id": "TKT-013", "ticket_text": "I returned the defective laptop from order ORD-2001 over three weeks ago, warehouse confirmed receipt, yet my $1,200 refund is still missing!", "category": "Refund", "order_id": "ORD-2001", "priority": "High"},
    {"id": "TKT-014", "ticket_text": "You cancelled my order ORD-2002 due to out-of-stock items five days ago, and I urgently need that money refunded to pay my rent.", "category": "Refund", "order_id": "ORD-2002", "priority": "High"},
    {"id": "TKT-015", "ticket_text": "I received a shattered monitor in order ORD-2003 and need an immediate full refund issued back to my original card today.", "category": "Refund", "order_id": "ORD-2003", "priority": "High"},
    {"id": "TKT-016", "ticket_text": "My return request for order ORD-2004 was wrongly rejected even though the item is unopened. Please escalate my refund immediately.", "category": "Refund", "order_id": "ORD-2004", "priority": "High"},
    {"id": "TKT-017", "ticket_text": "I sent back the wrong-sized jacket from order ORD-2005 on Monday. How long does it usually take to process the refund?", "category": "Refund", "order_id": "ORD-2005", "priority": "Medium"},
    {"id": "TKT-018", "ticket_text": "I would like to return two items from order ORD-2006 and get a refund instead of store credit. Please send me the return label.", "category": "Refund", "order_id": "ORD-2006", "priority": "Medium"},
    {"id": "TKT-019", "ticket_text": "The refund amount credited for order ORD-2007 is $15 less than what I paid. Can you review and reimburse the difference?", "category": "Refund", "order_id": "ORD-2007", "priority": "Medium"},
    {"id": "TKT-020", "ticket_text": "Please cancel order ORD-2008 before dispatch and refund the full purchase amount to my PayPal account.", "category": "Refund", "order_id": "ORD-2008", "priority": "Medium"},
    {"id": "TKT-021", "ticket_text": "What is your general return and refund window for seasonal clothing items?", "category": "Refund", "order_id": None, "priority": "Low"},
    {"id": "TKT-022", "ticket_text": "Just checking if my refund for order ORD-2010 has been initiated yet; whenever you have a moment to look.", "category": "Refund", "order_id": "ORD-2010", "priority": "Low"},
    {"id": "TKT-023", "ticket_text": "Are original shipping costs refundable if I decide to return an unwanted item?", "category": "Refund", "order_id": None, "priority": "Low"},
    {"id": "TKT-024", "ticket_text": "Can you confirm which card will receive the refund once my returned headphones from order ORD-2012 arrive?", "category": "Refund", "order_id": "ORD-2012", "priority": "Low"},

    # ==================== PAYMENT (12: 4 High, 4 Medium, 4 Low) ====================
    {"id": "TKT-025", "ticket_text": "Emergency! My credit card was charged three times for order ORD-3001 and my bank account is now overdrawn.", "category": "Payment", "order_id": "ORD-3001", "priority": "High"},
    {"id": "TKT-026", "ticket_text": "I see an unauthorized charge of $450 on my statement from your store even though I placed no order today. Please freeze and investigate immediately!", "category": "Payment", "order_id": None, "priority": "High"},
    {"id": "TKT-027", "ticket_text": "My payment for order ORD-3003 was deducted from my bank account, but your checkout page failed and says unpaid.", "category": "Payment", "order_id": "ORD-3003", "priority": "High"},
    {"id": "TKT-028", "ticket_text": "Our corporate procurement payment for order ORD-3004 is blocked at gateway verification and the supplier deadline expires in one hour.", "category": "Payment", "order_id": "ORD-3004", "priority": "High"},
    {"id": "TKT-029", "ticket_text": "My Visa card keeps getting declined at checkout with error code 402 when I try to pay for order ORD-3005.", "category": "Payment", "order_id": "ORD-3005", "priority": "Medium"},
    {"id": "TKT-030", "ticket_text": "I forgot to apply my 20% discount promo code during checkout for order ORD-3006. Can you adjust the billing total?", "category": "Payment", "order_id": "ORD-3006", "priority": "Medium"},
    {"id": "TKT-031", "ticket_text": "Can I switch the payment method on my pending subscription renewal from my expired card to a new Mastercard?", "category": "Payment", "order_id": None, "priority": "Medium"},
    {"id": "TKT-032", "ticket_text": "Why is there an extra international transaction fee listed on the receipt for order ORD-3008?", "category": "Payment", "order_id": "ORD-3008", "priority": "Medium"},
    {"id": "TKT-033", "ticket_text": "Do you accept Apple Pay, Google Pay, or bank wire transfers for online purchases?", "category": "Payment", "order_id": None, "priority": "Low"},
    {"id": "TKT-034", "ticket_text": "Could you email me a PDF copy of the VAT tax invoice for order ORD-3010 for my accounting records?", "category": "Payment", "order_id": "ORD-3010", "priority": "Low"},
    {"id": "TKT-035", "ticket_text": "Is it possible to split a payment across two different debit cards on your website?", "category": "Payment", "order_id": None, "priority": "Low"},
    {"id": "TKT-036", "ticket_text": "When will the pending authorization hold on my card for order ORD-3012 drop off?", "category": "Payment", "order_id": "ORD-3012", "priority": "Low"},

    # ==================== TECHNICAL (12: 4 High, 4 Medium, 4 Low) ====================
    {"id": "TKT-037", "ticket_text": "Critical issue: our entire team is locked out of the admin dashboard with a 500 Internal Server Error and production operations are halted!", "category": "Technical", "order_id": None, "priority": "High"},
    {"id": "TKT-038", "ticket_text": "The mobile app crashes immediately on launch after today's security update, preventing me from accessing my digital ticket for order ORD-4002.", "category": "Technical", "order_id": "ORD-4002", "priority": "High"},
    {"id": "TKT-039", "ticket_text": "Two-factor authentication SMS codes are not arriving at all, so I am completely locked out of my account.", "category": "Technical", "order_id": None, "priority": "High"},
    {"id": "TKT-040", "ticket_text": "The downloadable software license key I purchased in order ORD-4004 throws an invalid activation signature error on startup.", "category": "Technical", "order_id": "ORD-4004", "priority": "High"},
    {"id": "TKT-041", "ticket_text": "The password reset email link redirects me to a broken 404 page whenever I click it on Chrome.", "category": "Technical", "order_id": None, "priority": "Medium"},
    {"id": "TKT-042", "ticket_text": "Uploading profile photos fails with a timeout message if the image is larger than 2 MB.", "category": "Technical", "order_id": None, "priority": "Medium"},
    {"id": "TKT-043", "ticket_text": "I cannot download the firmware zip file included with my hardware purchase in order ORD-4007; the download stops at 50%.", "category": "Technical", "order_id": "ORD-4007", "priority": "Medium"},
    {"id": "TKT-044", "ticket_text": "Search filters on the product catalog page reset every time I navigate back from an item view.", "category": "Technical", "order_id": None, "priority": "Medium"},
    {"id": "TKT-045", "ticket_text": "Dark mode text on the settings page has slightly low contrast on Safari.", "category": "Technical", "order_id": None, "priority": "Low"},
    {"id": "TKT-046", "ticket_text": "Which web browsers and operating system versions are officially supported by your desktop client?", "category": "Technical", "order_id": None, "priority": "Low"},
    {"id": "TKT-047", "ticket_text": "How do I enable email notifications in my user profile preferences?", "category": "Technical", "order_id": None, "priority": "Low"},
    {"id": "TKT-048", "ticket_text": "The product manual PDF link for order ORD-4012 opens slowly on mobile data, just letting you know.", "category": "Technical", "order_id": "ORD-4012", "priority": "Low"},

    # ==================== OTHER (12: 4 High, 4 Medium, 4 Low) ====================
    {"id": "TKT-049", "ticket_text": "I suspect someone compromised my account and changed my registered email address. Please lock my profile immediately for security!", "category": "Other", "order_id": None, "priority": "High"},
    {"id": "TKT-050", "ticket_text": "Urgent GDPR data deletion request: please erase all my personal records immediately under Article 17 and confirm compliance today.", "category": "Other", "order_id": None, "priority": "High"},
    {"id": "TKT-051", "ticket_text": "We received a legal compliance notice regarding counterfeit listings using our trademark on your marketplace and need immediate executive contact.", "category": "Other", "order_id": None, "priority": "High"},
    {"id": "TKT-052", "ticket_text": "A safety recall alert was mentioned in the news for the battery model in my order ORD-5004; please tell me urgently if my unit is affected.", "category": "Other", "order_id": "ORD-5004", "priority": "High"},
    {"id": "TKT-053", "ticket_text": "How can our company apply to become an authorized wholesale distributor for your brand?", "category": "Other", "order_id": None, "priority": "Medium"},
    {"id": "TKT-054", "ticket_text": "I would like to update my account ownership details from my personal name to my registered business entity.", "category": "Other", "order_id": None, "priority": "Medium"},
    {"id": "TKT-055", "ticket_text": "Can you merge my two duplicate customer loyalty accounts so my reward points are combined?", "category": "Other", "order_id": None, "priority": "Medium"},
    {"id": "TKT-056", "ticket_text": "I want to submit a formal complaint regarding rude behavior from a phone support representative earlier this afternoon.", "category": "Other", "order_id": None, "priority": "Medium"},
    {"id": "TKT-057", "ticket_text": "Thank you so much to your support team for resolving my issue yesterday, everything works great now!", "category": "Other", "order_id": None, "priority": "Low"},
    {"id": "TKT-058", "ticket_text": "What are your customer service opening hours during public holidays?", "category": "Other", "order_id": None, "priority": "Low"},
    {"id": "TKT-059", "ticket_text": "Suggestion: it would be nice if you added a wishlist sharing button for gift registries.", "category": "Other", "order_id": None, "priority": "Low"},
    {"id": "TKT-060", "ticket_text": "Where is your company headquarters located and do you have any physical retail stores?", "category": "Other", "order_id": None, "priority": "Low"},
]

def main():
    output_path = Path("data/raw_tickets.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(RAW_TICKETS, f, indent=2, ensure_ascii=False)
    print(f"Successfully created {output_path} with {len(RAW_TICKETS)} raw tickets.")

if __name__ == "__main__":
    main()

