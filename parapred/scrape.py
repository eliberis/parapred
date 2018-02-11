from lxml import html
import requests


def download_annotated_seq(pdb, hchain, lchain):
    hchain = hchain.capitalize()
    lchain = lchain.capitalize()

    page = requests.get('http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/StrViewer.php?str=' + pdb)
    tree = html.fromstring(page.content)

    # Get FV info section for the correct pair of chains
    fv_info = tree.xpath("//section[@id='fv_info']")
    chains_info_div_id = fv_info[0].xpath(".//div[@class='accordion-heading']/a[contains(text(), '{} / {}')]/@href"
                                          .format(hchain, lchain))[0]
    chains_info = fv_info[0].xpath(".//div[@id='{}']".format(chains_info_div_id[1:]))

    # Get section describing chain residues
    ab_seqs_div_id = chains_info[0].xpath(".//div[@class='accordion-heading']/a[contains(text(), "
                                          "'Chothia-numbered antibody sequences')]/@href")[0]

    ab_seqs = fv_info[0].xpath(".//div[@id='{}']".format(ab_seqs_div_id[1:]))
    chains = ab_seqs[0].xpath("div/table/tr")[1:]

    if hchain == lchain:
        lchain = lchain.lower()

    output = {}
    for c in chains:
        chain_id = hchain if c[0][0].text.strip() == "VH" else lchain
        aa_names = c[3].xpath("table/tr/th/text()")
        residues = c[3].xpath("table/tr/td/font/text()")
        chain = {}
        for n, r in zip(aa_names, residues):
            key = extract_number_and_letter(n.strip())
            chain[key] = str(r)
        output[chain_id] = chain

    return output


def extract_number_and_letter(label):
    label = label[1:]  # Discard H or L
    letter = ""
    if label[-1].isalpha():
        letter = label[-1]
        label = label[:-1]
    return int(label), letter


if __name__ == "__main__":
    print(download_annotated_seq("2ghw", "B", "b"))
