"""
Synthetic Training Data Generator for Invoice Expense Classification

Generates 350+ realistic invoice text samples across 6 expense categories
with India-specific business terminology (GST, HSN, Indian courier brands,
domestic utilities, etc.) to match NextBill's MSME market.

Usage:
    python -m data.generate_data
"""

import os
import random
import pandas as pd

# ---------------------------------------------------------------------------
# Base invoice descriptions per category (20 each)
# ---------------------------------------------------------------------------

CATEGORY_TEMPLATES: dict[str, list[str]] = {

    "Logistics": [
        "Blue Dart courier charges for warehouse delivery",
        "DTDC express parcel delivery charges",
        "Delhivery shipment for customer order fulfilment",
        "FedEx international freight forwarding charges",
        "DHL express shipping for export consignment",
        "Ecom Express last-mile delivery fee",
        "India Post speed post and registered parcel charges",
        "XpressBees logistics pickup and delivery",
        "Shadowfax hyperlocal delivery charges",
        "Transport charges for inter-state goods movement",
        "Freight forwarding charges for container shipment",
        "Warehouse to warehouse stock transfer logistics",
        "Cold chain transportation for perishable goods",
        "Courier charges for sending documents to client",
        "Trucking charges for bulk raw material delivery",
        "Porter intra-city logistics and tempo hire",
        "Loading and unloading charges at distribution centre",
        "Return shipment reverse logistics processing fee",
        "E-way bill compliant transporter charges for goods",
        "Bike delivery charges for same-day dispatch",
    ],

    "Office Supplies": [
        "A4 printing paper ream pack for office use",
        "HP ink cartridge and toner replacement",
        "Desk organizers and file folders purchase",
        "Whiteboard markers and eraser set for meeting room",
        "Ergonomic office chair for workstation",
        "Standing desk converter and monitor riser",
        "Staplers paper clips and binder accessories",
        "Visiting cards and letterhead printing",
        "Office pantry supplies tea coffee sugar and cups",
        "Cleaning supplies and housekeeping materials",
        "Wall clock and notice board for reception",
        "UPS battery backup for office computers",
        "Projector screen for conference room",
        "Multi-plug extension board and cable organizer",
        "Employee ID card printing and lanyards",
        "Paper shredder for document disposal",
        "Keyboard and mouse combo wireless set",
        "Desk lamp LED study light for workstation",
        "Pen drive and external hard disk purchase",
        "First aid kit and safety supplies for office",
    ],

    "Cloud/Software": [
        "AWS monthly cloud hosting and EC2 instance bill",
        "Google Cloud Platform compute engine subscription",
        "Microsoft Azure virtual machine and storage charges",
        "Slack Business Plus annual team subscription",
        "Zoom Pro video conferencing license renewal",
        "GitHub Team plan for private repositories",
        "Jira Software Cloud project management license",
        "Notion team workspace annual subscription",
        "Figma professional design tool license",
        "Google Workspace Business email and storage plan",
        "Adobe Creative Cloud all-apps subscription",
        "Salesforce CRM monthly enterprise license",
        "DigitalOcean droplet and managed database charges",
        "Cloudflare domain and CDN service subscription",
        "MongoDB Atlas managed database hosting",
        "Vercel Pro deployment and hosting plan",
        "New Relic application monitoring subscription",
        "ChatGPT API usage and OpenAI platform credits",
        "Freshdesk customer support software license",
        "Razorpay payment gateway integration charges",
    ],

    "Utilities": [
        "Monthly electricity bill for office premises",
        "BSNL broadband internet connection charges",
        "Jio fiber high-speed internet monthly rental",
        "Airtel business Wi-Fi connection bill",
        "Piped water supply and sewerage charges",
        "Commercial gas connection bill for canteen",
        "MTNL telephone and landline rental charges",
        "Municipal property tax and civic charges",
        "Building maintenance and common area electricity",
        "Diesel generator fuel for power backup",
        "Fire safety compliance and extinguisher refill",
        "Pest control and fumigation service charges",
        "Lift and elevator annual maintenance contract",
        "CCTV surveillance system monitoring charges",
        "Solar panel maintenance and cleaning charges",
        "Air conditioning AMC and servicing charges",
        "Sewage treatment plant operating expenses",
        "Rain water harvesting system maintenance",
        "Waste disposal and garbage collection charges",
        "Street lighting and compound security charges",
    ],

    "Travel": [
        "Flight tickets Delhi to Mumbai for client meeting",
        "Rajdhani Express train fare for business travel",
        "OYO hotel accommodation for conference in Bangalore",
        "Ola cab ride from office to airport",
        "Uber auto rickshaw for local client visit",
        "Per diem meal allowance during Hyderabad trip",
        "Tatkal train booking for urgent site visit",
        "MakeMyTrip hotel booking for annual offsite",
        "Vistara business class ticket for CEO travel",
        "IRCTC train ticket for team offsite Jaipur",
        "Travel insurance premium for international trip",
        "Visa processing and consulate fees for Singapore",
        "Airport lounge access and priority boarding",
        "Fuel reimbursement for personal car used for work",
        "Bus ticket for employee inter-city commute",
        "Metro card recharge for daily office commute",
        "Parking charges at client premises and airport",
        "Toll charges for highway travel Pune to Mumbai",
        "Rental car with driver for week-long project site",
        "Conference registration and travel grant",
    ],

    "Inventory": [
        "Purchase of raw cotton bales from supplier",
        "Steel bars and TMT rods for manufacturing",
        "Electronic components ICs and PCBs for assembly",
        "Packaging boxes cartons and bubble wrap",
        "Finished goods stock replenishment from vendor",
        "Fabric rolls and textile material procurement",
        "Food ingredients and spices bulk purchase",
        "Pharmaceutical raw materials API procurement",
        "Automobile spare parts for service centre stock",
        "Mobile accessories wholesale lot for retail",
        "Cement bags and construction materials purchase",
        "Paint and chemical solvents for production",
        "Safety equipment helmets gloves for warehouse",
        "Garment trims buttons zippers thread stock",
        "Battery cells and charger units procurement",
        "Wood timber and plywood for furniture workshop",
        "Plastic granules and moulding material",
        "Printer consumables and toner for resale",
        "Herbal extracts and cosmetic raw materials",
        "Computer hardware components for assembly",
    ],
}

# ---------------------------------------------------------------------------
# Augmentation templates – wraps each base description in realistic contexts
# ---------------------------------------------------------------------------

AUGMENT_PATTERNS = [
    "{text}",
    "{text}",                                           # raw (2x weight)
    "Invoice for {text}",
    "Bill: {text}",
    "Payment towards {text}",
    "Vendor invoice — {text}",
    "GST invoice for {text}",
    "Expense: {text}",
    "{text} — reimbursement claim",
    "Recurring charge: {text}",
]


def generate_data(output_path: str | None = None) -> pd.DataFrame:
    """Generate synthetic training data and save to CSV."""

    rows: list[dict[str, str]] = []

    for category, templates in CATEGORY_TEMPLATES.items():
        for base_text in templates:
            for pattern in AUGMENT_PATTERNS:
                text = pattern.format(text=base_text)
                rows.append({"text": text, "category": category})

                # Add a lowercase variant for 30% of samples
                if random.random() < 0.3:
                    rows.append({"text": text.lower(), "category": category})

    # Shuffle
    random.seed(42)
    random.shuffle(rows)

    df = pd.DataFrame(rows)

    # Default output path
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "training_data.csv"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df)} training examples -> {output_path}")
    return df


if __name__ == "__main__":
    generate_data()
