book_info = {
    "title": "The Great Gatsby",
    "author": "F. Scott Fitzgerald",
    "publication_year": 1925,
    "genre": "Novel"
}
coordinates_info = {
    (40.7128, -74.0060,4): "New York City",
    (34.0522, -118.2437,5): "Los Angeles",
    (41.8781, -87.6298,6): "Chicago"
}

# Iterating through the dictionary to print keys and values together
for key, value in coordinates_info.items():
    print(f"Coordinates: {coordinates}, City: {city}")
