//! Suffix Automaton (SAM) for ROSA mechanism.
//!
//! This module implements a generic suffix automaton that supports incremental
//! token sequence processing and longest suffix matching queries.

use std::{collections::HashMap, hash::Hash};

use nonmax::NonMaxUsize;

/// Generic Suffix Automaton for ROSA pattern matching.
///
/// The suffix automaton efficiently processes discrete token sequences and supports
/// longest suffix matching queries in O(1) average time per operation.
///
/// # Type Parameters
/// - `T`: The token type, must be copyable, cloneable, and comparable.
pub struct Sam<T>
where
    T: Copy + Clone + PartialEq + Eq + Hash,
{
    /// All states in the automaton.
    states: Vec<State<T>>,
    /// The last state added.
    last: usize,
    /// The current sequence of tokens.
    sequence: Vec<T>,
}

/// A state in the suffix automaton.
#[derive(Debug, Clone)]
struct State<T> {
    /// Transition map: token → state index.
    next: HashMap<T, usize>,
    /// Suffix link (points to a state that represents a proper suffix).
    link: Option<NonMaxUsize>,
    /// Length of the longest string in this equivalence class.
    len: usize,
}

impl<T> Default for State<T> {
    fn default() -> Self {
        Self {
            next: HashMap::new(),
            link: None,
            len: 0,
        }
    }
}

impl<T> Sam<T>
where
    T: Copy + Clone + PartialEq + Eq + std::hash::Hash,
{
    /// Creates a new empty suffix automaton.
    pub fn new() -> Self {
        Self {
            states: vec![State::default()],
            last: 0,
            sequence: Vec::new(),
        }
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
    fn push_internal(&mut self, c: T) {
        let cur = self.states.len();
        self.states.push(State {
            next: HashMap::new(),
            link: None,
            len: self.states[self.last].len + 1,
        });

        // the (potential) first conflict state whose next state contains `c`
        let mut p = self.last;

        // add transitions from all suffix states that don't have transition on `c`
        while !self.states[p].next.contains_key(&c) {
            self.states[p].next.insert(c, cur);
            match self.states[p].link {
                Some(link) => p = link.into(),
                None => {
                    // reached the initial state
                    self.states[cur].link = NonMaxUsize::new(0);
                    self.last = cur;
                    return;
                }
            }
        }

        // the next state that `p` already transitions to on `c`
        let q = self.states[p].next[&c];

        if self.states[p].len + 1 == self.states[q].len {
            // simple case: `q` is the right suffix link
            self.states[cur].link = NonMaxUsize::new(q);
        } else {
            // need to clone state `q`
            let clone = self.states.len();
            let mut state = self.states[q].clone();
            state.len = self.states[p].len + 1;
            self.states.push(state);

            // update suffix link of `cur` and `q`
            self.states[cur].link = NonMaxUsize::new(clone);
            self.states[q].link = NonMaxUsize::new(clone);

            // redirect transitions from `p` and its suffixes
            while self.states[p].next.get(&c) == Some(&q) {
                self.states[p].next.insert(c, clone);
                match self.states[p].link {
                    Some(link) => p = link.into(),
                    None => break,
                }
            }
        }

        self.last = cur;
    }

    /// Returns true if the sequence contains the given pattern.
    pub fn contains(&self, pattern: &[T]) -> bool {
        if pattern.is_empty() {
            return true;
        }
        let mut current = 0;
        for &token in pattern {
            match self.states[current].next.get(&token) {
                Some(&next) => current = next,
                None => return false,
            }
        }
        true
    }
}

impl<T> Default for Sam<T>
where
    T: Copy + Clone + PartialEq + Eq + std::hash::Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let sam: Sam<char> = Sam::new();
        assert!(sam.is_empty());
        assert_eq!(sam.len(), 0);
    }

    #[test]
    fn test_push_and_extend() {
        let mut sam: Sam<char> = Sam::new();
        sam.push('a');
        assert_eq!(sam.len(), 1);

        sam.extend(&['b', 'c', 'd']);
        assert_eq!(sam.len(), 4);
        assert_eq!(sam.sequence(), &['a', 'b', 'c', 'd']);
    }

    #[test]
    fn test_contains() {
        let mut sam: Sam<char> = Sam::new();
        sam.extend(&['a', 'b', 'c', 'a', 'b', 'd']);

        // existing substrings
        assert!(sam.contains(&['a', 'b']));
        assert!(sam.contains(&['b', 'c']));
        assert!(sam.contains(&['c', 'a']));
        assert!(sam.contains(&['a', 'b', 'd']));
        assert!(sam.contains(&['a']));
        assert!(sam.contains(&['d']));
        assert!(sam.contains(&[])); // empty pattern

        // non-existing substrings
        assert!(!sam.contains(&['a', 'd']));
        assert!(!sam.contains(&['b', 'a']));
        assert!(!sam.contains(&['c', 'd']));
        assert!(!sam.contains(&['a', 'b', 'c', 'd'])); // Not a continuous substring
        assert!(!sam.contains(&['e']));
    }
}
