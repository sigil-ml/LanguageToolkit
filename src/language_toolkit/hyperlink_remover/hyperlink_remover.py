import re

# This pattern finds text of the form "[display](url)" and groups "url"
# Used for removal of Markdown formatting
PATTERN_MARKDOWN_REMOVAL = re.compile(r"""
    \[                  # Literal [
    (                   # Begin group
    [^\]]*              # Anything that is not a literal ], possibly repeated 
    )                   # Close group
    \]                  # Literal ]
    \(                  # Literal (
    [^\)]+              # Anything that is not a literal ) 
    \)                  # Literal )
    """, re.VERBOSE)

# This pattern finds links to Mattermost channels and groups the channel name
PATTERN_CHANNEL_NAME = re.compile(r"""
                          https://chat.il4.dso.mil/usaf-618aoc-mod/channels/
                          (
                          [0-9a-zA-Z_-]*
                          )
                          """, re.VERBOSE)

# This pattern finds any url
PATTERN_URL = re.compile(r"""
    (?:https?://|www\.)[^\s/$.?#].[^\s]*
    """, re.VERBOSE)

# This pattern finds a ~ followed by sufficiently long text
PATTERN_TILDE_TEXT = re.compile(r"""
                    ~
                    [^\s]{15,}
                     """, re.VERBOSE)

def sub_with_group(text: str, pattern: re.Pattern) -> str:
    '''
    Purpose: This function finds a pattern in text and replaces it with a sub-pattern
        as defined by a group in the regular expression. If there are multiple matches 
        in the text, each is replaced with its own corresponding sub-pattern. 

        In particular, it will be used to clean text from Mattermost, by
        extracting a label from Markdown format, and extracting a channel_id from a URL.

    Arguments: 
        - text: a string of text to be searched and have a portion substituted
        - pattern: a regular expression with at least one group.  Only the first
            group is used.
        
    Returns: a string with the original match replaced by the group in the pattern.

    Example:
        # Extracts a first and last name, with a group for the last name
        pattern = re.compile(r"""[A-Z][a-z]+\s([A-Z][a-z])""")
        text = 'Dalton Walker assigned Steve Carden and Leslie Douglas to this project.'
        sub_with_group(text, pattern)
        'Walker assigned Carden and Douglas to this project.'
    '''

    def replacement_function(match):
        # This is a helper function that extracts the first group from the match
        return match.group(1)
    
    return re.sub(pattern, replacement_function, text)