import unittest
import numpy as np
import data_provider
import structure_processor
import model

class TestDataProvider(unittest.TestCase):
    def setUp(self):
        dataset_desc_file = data_provider.TEST_DATASET_DESC_FILE
        self.chains = data_provider.load_chains(dataset_desc_file)

    def test_loading_chains(self):
        first_entry = next(self.chains)

        ag, abh, abl, name = first_entry

        self.assertEqual(name, "1ahw")
        self.assertEqual(len(ag), 200)
        self.assertEqual(len(abh), 214)
        self.assertEqual(len(abl), 214)

        # 29 entries remaining
        self.assertEqual(sum(1 for _ in self.chains), 29)

    def test_process_chains(self):
        ag, abh, abl, _ = next(self.chains)
        encoded = data_provider.process_chains(ag, abh, abl, 15, 200)

        ag_mat, cdrs, lbls, masks, class_split = encoded

        self.assertEqual(ag_mat.shape, (6, 200, 28))
        self.assertEqual(cdrs.shape, (6, 15, 28))
        self.assertEqual(lbls.shape, (6, 15, 1))
        self.assertEqual(masks.shape, (6, 15, 1))
        self.assertEqual(class_split, (23, 72))


class TestStructureProcessor(unittest.TestCase):
    def setUp(self):
        dataset_desc_file = data_provider.TEST_DATASET_DESC_FILE
        self.chains_1ahw = next(data_provider.load_chains(dataset_desc_file))

    def test_cdr_extraction_h(self):
        _, abh, _, _ = self.chains_1ahw
        cdrs = structure_processor.extract_cdrs(abh, ["H1", "H2", "H3"])
        self.assertEqual(len(cdrs["H1"]), 11)
        self.assertEqual(len(cdrs["H2"]), 10)
        self.assertEqual(len(cdrs["H3"]), 12)

    def test_cdr_extraction_l(self):
        _, _, abl, _ = self.chains_1ahw
        cdrs = structure_processor.extract_cdrs(abl, ["L1", "L2", "L3"])
        self.assertEqual(len(cdrs["L1"]), 15)
        self.assertEqual(len(cdrs["L2"]), 11)
        self.assertEqual(len(cdrs["L3"]), 13)

    def test_residue_seq_to_one(self):
        ag, _, _, _ = self.chains_1ahw

        ag_seq = "".join(structure_processor.residue_seq_to_one(ag))
        expected = \
            "TNTVAAYNLTWKSTNFKTILEWEPKPVNQVYTVQISTKSGDWKSKCFYTTDTECDLTDEI" \
            "VKDVKQTYLARVFSYPAGNEPLYENSPEFTPYLETNLGQPTIQSFEQVGTKVNVTVEDER" \
            "TLVRRNNTFLSLRDVFGKDLIYTLYYWKSSSSGKKTAKTNTNEFLIDVDKGENYCFSVQA" \
            "VIPSRTVNRKSTDSPVECMG"
        self.assertEqual(ag_seq, expected)

    def test_seq_to_one_hot(self):
        seq = "TNT"

        encoded = structure_processor.seq_to_one_hot(seq)

        expected = np.array(
            [[ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
             [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               1.60, 0.13, 2.95, -0.6, 6.52, 0.21, 0.22],
             [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36]])

        self.assertTrue(np.array_equal(encoded, expected))


class TestModels(unittest.TestCase):
    def test_main_model_compilation(self):
        m = model.get_model(max_ag_len=100, max_cdr_len=50)
        self.assertEqual(m.input_shape,
                          [(None, 100, 28), (None, 50, 28), (None, 50)])
        self.assertEqual(m.output_shape, (None, 50, 1))

    def test_baseline_model_compilation(self):
        m = model.baseline_model(max_ag_len=100, max_cdr_len=50)
        self.assertEqual(m.input_shape, (None, 50, 28))
        self.assertEqual(m.output_shape, (None, 50, 1))

    def test_ab_only_model_compilation(self):
        m = model.ab_only_model(max_ag_len=100, max_cdr_len=50)
        self.assertEqual(m.input_shape,
                          [(None, 50, 28), (None, 50)])
        self.assertEqual(m.output_shape, (None, 50, 1))

    def test_no_neighbourhood_model_compilation(self):
        m = model.no_neighbourhood_model(max_ag_len=100, max_cdr_len=50)
        self.assertEqual(m.input_shape,
                          [(None, 100, 28), (None, 50, 28), (None, 50)])
        self.assertEqual(m.output_shape, (None, 50, 1))

if __name__ == '__main__':
    unittest.main()
