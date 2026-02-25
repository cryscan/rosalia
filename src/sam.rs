//! Suffix Automaton (SAM) for ROSA mechanism.
//!
//! This module implements a generic suffix automaton that supports incremental
//! token sequence processing and longest suffix matching queries.

use nonmax::NonMaxUsize;

/// Generic Suffix Automaton for ROSA pattern matching.
///
/// The suffix automaton efficiently processes discrete token sequences and supports
/// longest suffix matching queries in O(1) average time per operation.
///
/// # Type Parameters
/// - `T`: The token type, must be copyable, cloneable, and comparable.
/// - `S`: The size of the vocabulary, must be a power of 2.
#[derive(Debug, Clone)]
pub struct Sam<T, const S: usize = 16> {
    /// All states in the automaton.
    states: Vec<State<S>>,
    /// The last state added.
    last: usize,
    /// The current sequence of tokens.
    sequence: Vec<T>,
}

/// A state in the suffix automaton.
#[derive(Debug, Clone)]
struct State<const S: usize> {
    /// Transition map: token → state index.
    next: Box<[Option<NonMaxUsize>; S]>,
    /// Suffix link (points to a state that represents a proper suffix).
    link: Option<NonMaxUsize>,
    /// Length of the longest string in this equivalence class.
    len: usize,
    /// The last position of the state occurrence in the sequence.
    end: Option<NonMaxUsize>,
}

impl<const S: usize> Default for State<S> {
    fn default() -> Self {
        Self {
            next: Box::new([None; S]),
            link: None,
            len: 0,
            end: None,
        }
    }
}

impl<T, const S: usize> Sam<T, S>
where
    T: Copy + Clone + PartialEq + Eq + Into<usize>,
{
    /// Creates a new empty suffix automaton.
    pub fn new() -> Self {
        Default::default()
    }

    /// Creates a new empty suffix automaton with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut sam = Self {
            states: vec![State::default()],
            last: 0,
            sequence: Vec::with_capacity(capacity),
        };
        sam.states.reserve(2 * capacity);
        sam
    }

    /// Returns the current length of the sequence.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Returns true if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Returns the current sequence.
    pub fn sequence(&self) -> &[T] {
        &self.sequence
    }

    /// Clears the automaton and resets it to initial state.
    pub fn clear(&mut self) {
        self.states = vec![State::default()];
        self.states[0].len = 0;
        self.states[0].link = None;
        self.last = 0;
        self.sequence.clear();
    }

    /// Adds a single token to the automaton.
    pub fn push(&mut self, token: T) {
        self.sequence.push(token);
        self.push_internal(token);
    }

    /// Adds multiple tokens to the automaton.
    pub fn extend(&mut self, tokens: &[T]) {
        for &token in tokens {
            self.push(token);
        }
    }

    /// Internal method to extend the automaton with a single token.
    fn push_internal(&mut self, token: T) {
        let current = self.states.len();
        self.states.push(State {
            link: None,
            len: self.states[self.last].len + 1,
            ..Default::default()
        });

        // the (potential) first conflict state whose next state contains `token`
        let mut p = self.last;
        let token = token.into();

        // add transitions from all suffix states that don't have transition on `token`
        while self.states[p].next[token].is_none() {
            self.states[p].next[token] = NonMaxUsize::new(current);
            match self.states[p].link.map(Into::into) {
                Some(link) => p = link,
                None => {
                    // reached the initial state
                    self.states[current].link = NonMaxUsize::new(0);
                    self.states[current].end = NonMaxUsize::new(current);
                    self.last = current;
                    return;
                }
            }
        }

        // the next state that `p` already transitions to on `token`
        let q: usize = self.states[p].next[token]
            .expect("`q` must be valid")
            .into();

        if self.states[p].len + 1 == self.states[q].len {
            // simple case: `q` is the right suffix link
            self.states[current].link = NonMaxUsize::new(q);
        } else {
            // need to clone state `q`
            let clone = self.states.len();
            let mut state = self.states[q].clone();
            state.len = self.states[p].len + 1;
            self.states.push(state);

            // update suffix link of `current` and `q`
            self.states[current].link = NonMaxUsize::new(clone);
            self.states[q].link = NonMaxUsize::new(clone);

            // redirect transitions from `p` and its suffixes
            while self.states[p].next[token] == NonMaxUsize::new(q) {
                self.states[p].next[token] = NonMaxUsize::new(clone);
                match self.states[p].link.map(Into::into) {
                    Some(link) => p = link,
                    None => break,
                }
            }
        }

        // update end position of all suffixes on the chain
        let mut p = current;
        while let Some(link) = self.states[p].link.map(Into::into) {
            self.states[p].end = NonMaxUsize::new(current);
            p = link;
        }

        self.last = current;
    }

    /// Returns true if the sequence contains the given pattern.
    pub fn contains(&self, pattern: &[T]) -> bool {
        if pattern.is_empty() {
            return true;
        }
        let mut p = 0;
        for &token in pattern {
            let token = token.into();
            match self.states[p].next[token].map(Into::into) {
                Some(next) => p = next,
                None => return false,
            }
        }
        true
    }

    /// Returns the end position of the last match.
    pub fn match_end(&self, pattern: &[T]) -> Option<usize> {
        if pattern.is_empty() {
            return Some(0);
        }
        let mut p = 0;
        for &token in pattern {
            let token = token.into();
            match self.states[p].next[token].map(Into::into) {
                Some(next) => p = next,
                None => return None,
            }
        }
        self.states[p].end.map(Into::into)
    }

    /// Given a cached start `state`, match a new input `token`.
    ///
    /// # Returns
    ///
    /// `(Some(end), state)` if the match is successful, `(None, 0)` otherwise.
    pub fn match_end_incremental(&self, token: T, state: usize) -> (Option<usize>, usize) {
        let token = token.into();

        let mut p = state;
        while self.states[p].next[token].is_none()
            && let Some(link) = self.states[p].link.map(Into::into)
        {
            p = link;
        }

        match self.states[p].next[token].map(usize::from) {
            Some(next) => (self.states[next].end.map(Into::into), next),
            None => (None, 0),
        }
    }
}

impl<T, const S: usize> Default for Sam<T, S> {
    fn default() -> Self {
        Self {
            states: vec![State::default()],
            last: 0,
            sequence: Vec::new(),
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Rosa<T, V, const S: usize> {
    /// Query tokens.
    qs: Vec<T>,
    /// Key tokens form a SAM.
    ks: Sam<T, S>,
    /// Value tokens.
    vs: Vec<V>,
    /// Current state of longest matched `qs` suffix in `ks`.
    current: usize,
    /// Current len of the match.
    len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(unused)]
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, variant_count::VariantCount)]
    enum Token {
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
    }

    impl From<Token> for usize {
        fn from(token: Token) -> usize {
            token as usize
        }
    }

    #[test]
    fn test_basic_construction() {
        let sam = Sam::<Token, { Token::VARIANT_COUNT }>::new();
        assert!(sam.is_empty());
        assert_eq!(sam.len(), 0);
    }

    #[test]
    fn test_push_and_extend() {
        use Token::*;

        let mut sam = Sam::<Token, { Token::VARIANT_COUNT }>::new();
        sam.push(A);
        assert_eq!(sam.len(), 1);

        sam.extend(&[B, C, D]);
        assert_eq!(sam.len(), 4);
        assert_eq!(sam.sequence(), &[A, B, C, D]);
    }

    #[test]
    fn test_contains() {
        use Token::*;

        let mut sam = Sam::<Token, { Token::VARIANT_COUNT }>::new();
        sam.extend(&[A, B, C, A, B, D]);

        // existing substrings
        assert!(sam.contains(&[A, B]));
        assert!(sam.contains(&[B, C]));
        assert!(sam.contains(&[C, A]));
        assert!(sam.contains(&[A, B, D]));
        assert!(sam.contains(&[A]));
        assert!(sam.contains(&[D]));
        assert!(sam.contains(&[])); // empty pattern

        // non-existing substrings
        assert!(!sam.contains(&[A, D]));
        assert!(!sam.contains(&[B, A]));
        assert!(!sam.contains(&[C, D]));
        assert!(!sam.contains(&[A, B, C, D])); // Not a continuous substring
        assert!(!sam.contains(&[E]));
    }

    #[test]
    fn test_match_end() {
        use Token::*;

        let mut sam = Sam::<Token, { Token::VARIANT_COUNT }>::new();
        sam.extend(&[A, B, C, A, B, D]);

        // empty pattern should return Some(0)
        assert_eq!(sam.match_end(&[]), Some(0));

        // test patterns that exist and verify their end positions
        // pattern "AB" appears at positions 1 and 4
        // the last occurrence ends at position 4 (0-indexed)
        assert_eq!(sam.match_end(&[A, B]), Some(5));

        // pattern "BC" appears at position 2
        assert_eq!(sam.match_end(&[B, C]), Some(3));

        // pattern "CA" appears at position 3
        assert_eq!(sam.match_end(&[C, A]), Some(4));

        // pattern "ABD" appears at position 5
        assert_eq!(sam.match_end(&[A, B, D]), Some(6));

        // single token patterns
        assert_eq!(sam.match_end(&[A]), Some(4));
        assert_eq!(sam.match_end(&[B]), Some(5));
        assert_eq!(sam.match_end(&[C]), Some(3));
        assert_eq!(sam.match_end(&[D]), Some(6));

        // non-existing patterns should return None
        assert_eq!(sam.match_end(&[A, D]), None);
        assert_eq!(sam.match_end(&[B, A]), None);
        assert_eq!(sam.match_end(&[C, D]), None);
        assert_eq!(sam.match_end(&[A, B, C, D]), None);
        assert_eq!(sam.match_end(&[E]), None);
    }

    #[test]
    fn test_match_end_incremental() {
        use Token::*;

        let mut sam = Sam::<Token, { Token::VARIANT_COUNT }>::new();

        // start from initial state (0), match 'A'
        sam.push(A);
        let (end, state) = sam.match_end_incremental(A, 0);
        assert_eq!(end, Some(1));
        log::info!("state: {state}");

        // continue matching 'B'
        sam.push(B);
        let (end, state) = sam.match_end_incremental(B, state);
        assert_eq!(end, Some(2));
        log::info!("state: {state}");

        // matches "ABA"
        sam.extend(&[C, A, B, A]);
        let (end, state) = sam.match_end_incremental(A, state);
        assert_eq!(end, Some(6));
        log::info!("state: {state}");

        // non-existing pattern should return None
        sam.push(E);
        let (end, state) = sam.match_end_incremental(D, state);
        assert_eq!(end, None);
        log::info!("state: {state}");

        // continue matching 'E'
        sam.extend(&[F, G, H]);
        let (end, state) = sam.match_end_incremental(E, state);
        assert_eq!(end, Some(7));
        log::info!("state: {state}");

        // continue matching "EF"
        let (end, state) = sam.match_end_incremental(F, state);
        assert_eq!(end, Some(8));
        log::info!("state: {state}");

        // continue matching "EFG"
        let (end, state) = sam.match_end_incremental(G, state);
        assert_eq!(end, Some(9));
        log::info!("state: {state}");

        // continue matching "EFGH"
        let (end, state) = sam.match_end_incremental(H, state);
        assert_eq!(end, Some(10));
        log::info!("state: {state}");

        // key is "ABCABAEFGH", query is "ABADEFGHA", matches 'A'
        let (end, state) = sam.match_end_incremental(A, state);
        assert_eq!(end, Some(6));
        log::info!("state: {state}");

        // matches "AB"
        let (end, state) = sam.match_end_incremental(B, state);
        assert_eq!(end, Some(5));
        log::info!("state: {state}");

        // matches "ABC"
        let (end, state) = sam.match_end_incremental(C, state);
        assert_eq!(end, Some(3));
        log::info!("state: {state}");

        // key is "ABCABAEFGHABCD", query is "ABADEFGHABCD", matches "EFGHABCD"
        sam.extend(&[A, B, C, D]);
        let (end, state) = sam.match_end_incremental(D, state);
        assert_eq!(end, Some(14));
        log::info!("state: {state}");
    }
}
