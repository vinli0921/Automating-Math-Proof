import re
from lxml import etree
import mwparserfromhell
import pandas as pd
import os
import sys
from typing import Dict, List, Optional

# Define namespace mapping
NS = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

def parse_xml(xml_file_path: str):
    """Parses the XML file and yields each <page> element."""
    print(f"Starting to parse XML: {xml_file_path}")
    try:
        context = etree.iterparse(xml_file_path, events=('end',), 
                                tag='{http://www.mediawiki.org/xml/export-0.11/}page')
    except Exception as e:
        print(f"Error initializing iterparse: {e}")
        sys.exit(1)
    
    page_count = 0
    for _, elem in context:
        page_count += 1
        if page_count <= 5:
            title_elem = elem.find('mw:title', namespaces=NS)
            title = title_elem.text if title_elem is not None else 'No Title'
            print(f"Processing Page {page_count}: {title}")
        yield elem
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context
    print(f"Total pages processed in parse_xml: {page_count}")

def is_relevant_page(title: str, namespace: str) -> Optional[str]:
    """Determines if a page is an Axiom, Definition or proof."""
    if namespace not in ['0', '100', '102']:
        return None

    title_patterns = {
        'def': re.compile(r'^(Definition)\s*:', re.IGNORECASE),
        'axiom': re.compile(r'^(Axiom)\s*:', re.IGNORECASE),
    }

    for key, pattern in title_patterns.items():
        if pattern.match(title):
            return key
        return 'proof'

    return None

def extract_content(wikitext: str, heading_title: str) -> str:
    """Removes content after a specified heading (including the heading itself)."""
    wikicode = mwparserfromhell.parse(wikitext)
    sections = wikicode.get_sections(include_lead=True, include_headings=True)
    
    content_parts = []
    for section in sections:
        headings = section.filter_headings()
        if headings:
            heading = headings[0]
            heading_text = heading.title.strip_code().strip().lower()
            if heading_text == heading_title.lower():
                break
        content_parts.append(str(section))
    return ''.join(content_parts)

def extract_math_structure(content: str) -> Dict[str, str]:
    """Extracts mathematical structure from content."""
    if not isinstance(content, str):
        return {'theorem': '', 'proof': '', 'math_expressions': []}
        
    # Extract math expressions
    math_patterns = [
        r'\$.*?\$',  # Inline math
        r'\\ds.*?(?=\\ds|$)',  # Display style math
        r'\\begin\{.*?\}.*?\\end\{.*?\}'  # Environment math
    ]
    
    math_expressions = []
    for pattern in math_patterns:
        matches = re.finditer(pattern, content, re.DOTALL)
        math_expressions.extend(match.group(0) for match in matches)
    
    # Extract theorem and proof sections
    section_pattern = re.compile(r'==\s*(.*?)\s*==\s*(.*?)(?===\s*\w+\s*==|$)', 
                               re.DOTALL | re.IGNORECASE)
    
    sections = {
        'theorem': '',
        'proof': '',
        'math_expressions': math_expressions
    }
    
    for match in section_pattern.finditer(content):
        section_name = match.group(1).strip().lower()
        section_content = match.group(2).strip()
        if section_name == 'theorem':
            sections['theorem'] = section_content
        elif section_name == 'proof':
            sections['proof'] = section_content
            
    return sections

def determine_math_relationship(context: str, from_type: str, to_title: str) -> str:
    """Determine mathematical relationship type based on context and node types."""
    if to_title.startswith('Definition:'):
        if from_type == 'proof':
            return 'USES_DEFINITION'
        elif from_type == 'def':
            return 'RELATED_DEFINITION'
            
    elif to_title.startswith('Axiom:'):
        if re.search(r'using|by|requires|from', context.lower()):
            return 'USES_AXIOM'
            
    elif from_type == 'proof':
        proof_patterns = [
            (r'similar\s+to', 'SIMILAR_PROOF'),
            (r'follows\s+from', 'PROOF_DEPENDENCY'),
            (r'proof\s+(?:using|by)', 'PROOF_TECHNIQUE')
        ]
        
        for pattern, rel in proof_patterns:
            if re.search(pattern, context.lower()):
                return rel
    
    return 'LINK'

def collect_nodes(xml_file_path: str) -> pd.DataFrame:
    """Parses the XML and collects relevant nodes."""
    data = []
    total_pages = 0
    relevant_pages = 0
    redirect_pages = 0

    for page in parse_xml(xml_file_path):
        total_pages += 1

        # Extract basic page information
        title_elem = page.find('mw:title', namespaces=NS)
        title = title_elem.text if title_elem is not None else ''
        
        if '/Also known as' in title or '/Mistake' in title:
            continue

        ns_elem = page.find('mw:ns', namespaces=NS)
        namespace = ns_elem.text if ns_elem is not None else ''

        # Skip redirects
        redirect = page.find('mw:redirect', namespaces=NS)
        if redirect is not None:
            redirect_pages += 1
            continue

        # Check relevance
        node_type = is_relevant_page(title, namespace)
        if node_type:
            relevant_pages += 1

            # Extract content
            text_elem = page.find('mw:revision/mw:text', namespaces=NS)
            if text_elem is not None and text_elem.text:
                wikitext = text_elem.text
                content = extract_content(wikitext, 'Sources')
                
                # Extract mathematical structure
                math_structure = extract_math_structure(content)
                
                # Extract name from title
                name = title.split(':', 1)[1].strip() if ':' in title else title

                # Get node ID
                node_id_elem = page.find('mw:id', namespaces=NS)
                node_id = node_id_elem.text if node_id_elem is not None else None

                data.append({
                    'id': node_id,
                    'type': node_type,
                    'title': title,
                    'name': name,
                    'content': content,
                    'theorem': math_structure['theorem'],
                    'proof': math_structure['proof'],
                    'math_expressions': math_structure['math_expressions']
                })

                if relevant_pages <= 5:
                    print(f"Collected {node_type}: {name}")

        if total_pages % 1000 == 0:
            print(f"Processed {total_pages} pages, found {relevant_pages} relevant pages so far, skipped {redirect_pages} redirects.")

    df = pd.DataFrame(data)
    print(f"Total pages processed: {total_pages}")
    print(f"Total relevant pages found: {relevant_pages}")
    print(f"Total redirect pages skipped: {redirect_pages}")
    return df

def extract_relationships_from_links(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts typed relationships between nodes."""
    relationships = []
    
    # Build node lookup maps
    title_name_to_node = {}
    for _, row in nodes_df.iterrows():
        node_info = row.to_dict()
        if isinstance(row['title'], str):
            title_name_to_node[row['title'].lower()] = node_info
        if isinstance(row['name'], str):
            title_name_to_node[row['name'].lower()] = node_info
    
    for _, from_node in nodes_df.iterrows():
        content = from_node['content']
        if not isinstance(content, str):
            continue
            
        wikicode = mwparserfromhell.parse(content)
        for link in wikicode.filter_wikilinks():
            link_target = str(link.title).strip()
            link_target = re.split(r'[#|]', link_target)[0].strip()
            link_target_lower = link_target.lower()
            
            to_node = title_name_to_node.get(link_target_lower)
            if not to_node:
                continue
                
            # Get context around link
            text = str(wikicode)
            link_pos = text.find(str(link))
            context = text[max(0, link_pos-50):min(len(text), link_pos+len(str(link))+50)]
            
            rel_type = determine_math_relationship(
                context,
                from_node['type'],
                to_node['title']
            )
            
            relationships.append({
                'from_id': from_node['id'],
                'to_id': to_node['id'],
                'type': rel_type,
                'context': context
            })
    
    relationships_df = pd.DataFrame(relationships).drop_duplicates()
    print(f"Total relationships extracted: {len(relationships_df)}")
    print("Relationship types distribution:")
    print(relationships_df['type'].value_counts())
    return relationships_df

def save_to_csv(df: pd.DataFrame, filename: str):
    """Saves the DataFrame to a CSV file."""
    if not df.empty:
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved {len(df)} records to {filename}")
    else:
        print(f"No records to save for {filename}")

def main(xml_dump_path: str):
    """Main function to execute the extraction process."""
    if not os.path.isfile(xml_dump_path):
        print(f"Error: File '{xml_dump_path}' does not exist.")
        sys.exit(1)

    # Collect nodes
    nodes_df = collect_nodes(xml_dump_path)
    print(f"Number of nodes collected: {len(nodes_df)}")
    save_to_csv(nodes_df, 'nodes.csv')

    # Extract and save relationships
    if not nodes_df.empty:
        relationships_df = extract_relationships_from_links(nodes_df)
        save_to_csv(relationships_df, 'relationships.csv')
    else:
        print("No nodes to extract relationships from.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_proofs_updated.py /path/to/wikiproof_dump.xml")
        sys.exit(1)

    xml_dump_path = sys.argv[1]
    main(xml_dump_path)