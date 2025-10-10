from Bio.Align import PairwiseAligner


def compute_alignment_indices(sequences, ref_seq):
    """
    Align multiple sequences to a reference sequence.

    Returns:
        dict with:
            - 'aligned_seqs': list[str]  # aligned sequences (with gaps)
            - 'index_maps': list[list[Optional[int]]]  # aln_col -> seq_pos (None if gap)
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.target_open_gap_score = -10
    aligner.target_extend_gap_score = -10
    aligner.query_open_gap_score = -1
    aligner.query_extend_gap_score = -0.1

    aligned_seqs = []
    index_maps = []

    for seq in sequences:
        aln = aligner.align(ref_seq, seq)[0]

        # Reconstruct aligned strings from block coordinates
        ref_aln_chars = []
        seq_aln_chars = []
        ref_pos = 0
        seq_pos = 0
        ref_blocks, seq_blocks = aln.aligned

        for (r_start, r_end), (s_start, s_end) in zip(ref_blocks, seq_blocks):
            # gaps in ref before this block
            if r_start > ref_pos:
                ref_aln_chars.extend(ref_seq[ref_pos:r_start])
                seq_aln_chars.extend('-' * (r_start - ref_pos))
            # gaps in seq before this block
            if s_start > seq_pos:
                ref_aln_chars.extend('-' * (s_start - seq_pos))
                seq_aln_chars.extend(seq[seq_pos:s_start])

            # aligned block
            ref_aln_chars.extend(ref_seq[r_start:r_end])
            seq_aln_chars.extend(seq[s_start:s_end])

            ref_pos = r_end
            seq_pos = s_end

        # trailing tails
        if ref_pos < len(ref_seq):
            ref_aln_chars.extend(ref_seq[ref_pos:])
            seq_aln_chars.extend('-' * (len(ref_seq) - ref_pos))
        if seq_pos < len(seq):
            ref_aln_chars.extend('-' * (len(seq) - seq_pos))
            seq_aln_chars.extend(seq[seq_pos:])

        ref_aln = ''.join(ref_aln_chars)
        seq_aln = ''.join(seq_aln_chars)

        # Build alignment-column -> sequence-index map
        aln_to_seq_idx = []
        seq_non_gap_idx = 0
        for s in seq_aln:
            if s == '-':
                aln_to_seq_idx.append(None)
            else:
                aln_to_seq_idx.append(seq_non_gap_idx)
                seq_non_gap_idx += 1

        # Sanity check
        if len(aln_to_seq_idx) != len(seq_aln):
            raise RuntimeError("Index map length must match alignment length.")

        aligned_seqs.append(seq_aln)
        index_maps.append(aln_to_seq_idx)

    return {'aligned_seqs': aligned_seqs, 'index_maps': index_maps}