#!/usr/bin/env python3
"""
Test different month mappings for Merrill Edge options
"""

def test_month_mappings():
    """Test different month letter mappings"""
    
    # Test case: V1725D260000 = SOFI Put, $26.00 strike, Oct 17 2025
    ticker = "SOFIV1725D260000"
    
    print(f"🔍 Testing ticker: {ticker}")
    print(f"Expected: SOFI Put, $26.00 strike, Oct 17 2025")
    
    # Extract components
    underlying = "SOFI"
    month_letter = "V"
    day = "17"
    year = "25"
    decimal_code = "D"
    strike_digits = "260000"
    
    print(f"\n📊 Extracted components:")
    print(f"  Underlying: {underlying}")
    print(f"  Month letter: {month_letter}")
    print(f"  Day: {day}")
    print(f"  Year: {year}")
    print(f"  Decimal code: {decimal_code}")
    print(f"  Strike digits: {strike_digits}")
    
    # Test different month mappings
    print(f"\n🗓️ Testing month mappings:")
    
    # Mapping 1: Simple A=1, B=2, etc.
    month_num_simple = ord(month_letter) - ord('A') + 1
    print(f"  Simple A=1: V = month {month_num_simple}")
    
    # Mapping 2: Skip confusing letters (I, O, Q, etc.)
    confusing_letters = ['I', 'O', 'Q', 'U', 'V', 'W', 'X', 'Y', 'Z']
    month_num_skip = month_num_simple
    for letter in confusing_letters:
        if ord(letter) <= ord(month_letter):
            month_num_skip -= 1
    print(f"  Skip confusing: V = month {month_num_skip}")
    
    # Mapping 3: Custom mapping (need to figure out)
    custom_mapping = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
        'G': 7, 'H': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
        'N': 1, 'P': 2, 'R': 3, 'S': 4, 'T': 5, 'U': 6,
        'V': 7, 'W': 8, 'X': 9, 'Y': 10, 'Z': 11
    }
    month_num_custom = custom_mapping.get(month_letter, 0)
    print(f"  Custom mapping: V = month {month_num_custom}")
    
    # Test strike price calculation
    print(f"\n💰 Strike price calculation:")
    strike_int = int(strike_digits)
    if decimal_code == 'D':
        strike_price = float(strike_int) / 100
        print(f"  D{strike_digits} = ${strike_price}")
    
    # Test call/put determination
    print(f"\n📞 Call/Put determination:")
    print(f"  If V = month 7 (July): Call (A-L = 1-12)")
    print(f"  If V = month 19 (July): Put (M-X = 13-24)")
    print(f"  If V = month 7 (July): Call (A-L = 1-12)")

if __name__ == "__main__":
    test_month_mappings()
