from feature_metadata_generator import FeatureMetadataGenerator
# Create the generator
generator = FeatureMetadataGenerator('input/login_history_pc1_config.yaml')
results = generator.generate_and_validate_features()
print(f"Generated {results['total_features']} features")
# Generate feature metadata first
feature_metadata = generator.generate_feature_metadata()

# Generate the full batch query
batch_query = generator.generate_batch_query(feature_metadata)

# Print or save the query
print(batch_query)

# Save to file
with open('output/full_batch_query.sql', 'w') as f:
    f.write(batch_query)