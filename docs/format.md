# Coding Standards

## Documentation Comments (///)

- **Language**: Must use English
- **Capitalization**: The first letter of each sentence must be capitalized
- **Punctuation**: Must end with a period or other appropriate punctuation mark

### Examples

```rust
/// Calculates the sum of two numbers.
/// Returns the result as an integer.
fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Represents a user in the system.
/// Contains authentication information and preferences.
struct User {
    name: String,
    age: u32,
}
```

**Incorrect examples:**
```rust
/// calculates the sum  // Missing capitalization and punctuation
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

## Code Comments (//)

- **Language**: Must use English
- **Length**: Should not exceed one sentence
- **Capitalization**: Do not capitalize the first letter
- **Punctuation**: Do not use any punctuation at the end

### Examples

```rust
fn process_data(data: &str) {
    // trim whitespace from input
    let cleaned = data.trim();
    
    // check if empty
    if cleaned.is_empty() {
        return;
    }
    
    // convert to lowercase for consistency
    let lower = cleaned.to_lowercase();
    
    // split into tokens
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    
    for token in tokens {
        // process each token individually
        handle_token(token);
    }
}
```

**Incorrect examples:**
```rust
fn process_data(data: &str) {
    // Trim whitespace from input.  // Incorrect: capitalized and has punctuation
    // check if empty
    // Remove extra spaces and convert to lowercase  // Incorrect: more than one sentence
    if data.is_empty() {
        return;  // return if no data
    }
}
```

## Summary

| Comment Type | Language | Capitalization | Punctuation |
|--------------|----------|----------------|-------------|
| `///`        | English  | Yes            | Yes         |
| `//`         | English  | No             | No          |
