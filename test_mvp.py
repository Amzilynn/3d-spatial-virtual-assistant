import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

import spatial_assistant

class TestSpatialAssistant(unittest.TestCase):
    
    def test_scene_generation(self):
        print("\n[Test] Generaring Scene...")
        gen = spatial_assistant.SceneGenerator()
        pcd, objects = gen.create_synthetic_room()
        self.assertIsNotNone(pcd)
        self.assertEqual(len(objects), 3) # Sofa, Table, Chair
        print("  > Scene generated successfully.")

    def test_spatial_reasoning(self):
        print("\n[Test] Spatial Reasoning...")
        # Setup specific known objects
        objects = [
            {"label": "Sofa", "center": [0, 0, 0], "bounds": None},
            {"label": "Table", "center": [2, 0, 2], "bounds": None}
        ]
        engine = spatial_assistant.SpatialReasoningEngine(objects)
        
        # Mock user pos at (0,0,-2) -> Sofa is at (0,0,0) which is 2m in front
        engine.user_pos = np.array([0, 0, -2])
        engine.user_heading = np.array([0, 0, 1])

        ans, hl = engine.process_query("Where is the Sofa?")
        print(f"  > Query: 'Where is the Sofa?' -> Answer: '{ans}'")
        self.assertIn("Sofa", ans)
        self.assertIn("front", ans)
        self.assertEqual(hl, "Sofa")

    @patch('spatial_assistant.o3d.visualization.draw_geometries')
    @patch('builtins.input', side_effect=["Where is the chair?", "exit"])
    def test_cli_flow(self, mock_input, mock_draw):
        print("\n[Test] Interactive CLI Flow...")
        spatial_assistant.interactive_cli()
        print("  > CLI ran without error.")
        # Ensure visualizer was called at least once
        self.assertTrue(mock_draw.called)

if __name__ == '__main__':
    unittest.main()
