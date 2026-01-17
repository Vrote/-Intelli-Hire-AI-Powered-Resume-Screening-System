import numpy as np
from typing import List, Tuple, Dict

class DAAAlgorithms:
    """
    Implementation of DAA concepts:
    1. Dynamic Programming - Edit Distance (Levenshtein)
    2. String Matching - KMP Algorithm
    """
    
    # ========== DYNAMIC PROGRAMMING ==========
    
    @staticmethod
    def edit_distance(str1: str, str2: str) -> int:
        """
        Levenshtein Distance using Dynamic Programming
        
        Concept: Find minimum operations (insert/delete/replace) 
                 to transform str1 into str2
        
        Time Complexity: O(m × n)
        Space Complexity: O(m × n) can be optimized to O(min(m,n))
        
        Example:
        edit_distance("kitten", "sitting") = 3
        - kitten → sitten (substitute k→s)
        - sitten → sittin (substitute e→i)
        - sittin → sitting (insert g)
        """
        m, n = len(str1), len(str2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all characters from str1
        for j in range(n + 1):
            dp[0][j] = j  # Insert all characters to str1
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Delete from str1
                        dp[i][j-1],      # Insert to str1
                        dp[i-1][j-1]     # Replace in str1
                    )
        
        return dp[m][n]
    
    @staticmethod
    def similarity_score(str1: str, str2: str) -> float:
        """
        Convert edit distance to similarity score (0-100%)
        
        Formula: similarity = (1 - distance/max_length) × 100
        """
        str1, str2 = str1.lower(), str2.lower()
        distance = DAAAlgorithms.edit_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 100.0
        
        similarity = (1 - distance / max_len) * 100
        return round(similarity, 2)
    
    # ========== STRING MATCHING ==========
    
    @staticmethod
    def kmp_search(text: str, pattern: str) -> List[int]:
        """
        Knuth-Morris-Pratt (KMP) String Matching Algorithm
        
        Concept: Efficient pattern matching using failure function
                 Avoids re-comparing already matched characters
        
        Time Complexity: O(n + m) where n=text length, m=pattern length
        Space Complexity: O(m) for failure function
        
        Returns: List of starting indices where pattern is found
        
        Example:
        kmp_search("ababcababa", "aba") → [0, 5, 7]
        """
        if not pattern:
            return []
        
        text = text.lower()
        pattern = pattern.lower()
        
        # Build failure function (longest proper prefix which is also suffix)
        def compute_lps(pattern):
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        lps = compute_lps(pattern)
        matches = []
        
        i = 0  # text index
        j = 0  # pattern index
        n = len(text)
        m = len(pattern)
        
        while i < n:
            if text[i] == pattern[j]:
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and text[i] != pattern[j]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    @staticmethod
    def fuzzy_skill_match(resume_skills: List[str], required_skill: str, 
                         threshold: float = 70.0) -> Tuple[bool, str, float]:
        """
        Fuzzy matching using DP-based edit distance
        
        Combines:
        1. Exact match (KMP)
        2. Fuzzy match (Edit Distance)
        
        Returns: (is_match, matched_skill, similarity_score)
        """
        required_skill = required_skill.lower().strip()
        
        for resume_skill in resume_skills:
            resume_skill = resume_skill.lower().strip()
            
            # Method 1: Exact substring match using KMP
            if DAAAlgorithms.kmp_search(resume_skill, required_skill):
                return True, resume_skill, 100.0
            
            if DAAAlgorithms.kmp_search(required_skill, resume_skill):
                return True, resume_skill, 100.0
            
            # Method 2: Fuzzy match using Edit Distance
            similarity = DAAAlgorithms.similarity_score(resume_skill, required_skill)
            if similarity >= threshold:
                return True, resume_skill, similarity
        
        return False, "", 0.0
    
    @staticmethod
    def longest_common_subsequence(str1: str, str2: str) -> int:
        """
        LCS using Dynamic Programming
        
        Time Complexity: O(m × n)
        Space Complexity: O(m × n)
        
        Used for measuring text similarity
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]