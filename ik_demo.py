#!/usr/bin/env python3
"""Demo script showing inverse kinematics with robotic arm visualization."""

import numpy as np
import mujoco.viewer
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.testing import assets
from dm_control.utils import inverse_kinematics as ik
import time

mjlib = mjbindings.mjlib

# Constants
SITE_NAME = 'gripsite'
JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
TARGET_SPHERE_SIZE = 0.03
ANIMATION_STEPS = 100

class IKDemo:
    def __init__(self):
        # Target position and orientation
        self.target_pos = np.array([0.3, 0.2, 0.4])
        # Generate random initial orientation
        self.target_quat = np.random.randn(4)
        self.target_quat = self.target_quat / np.linalg.norm(self.target_quat)
        
        # Load the model directly from MJCF file
        arm_xml = assets.get_contents('test.mjcf').decode('utf-8')
        self.physics = mujoco.Physics.from_xml_string(arm_xml)
        
        # Set initial target pose
        self.physics.named.model.body_pos['target'] = self.target_pos
        self.physics.named.model.body_quat['target'] = self.target_quat
        self.physics.forward()
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
        
    def set_random_arm_configuration(self):
        """Set the arm to a random configuration."""
        # Get joint limits
        joint_ranges = self.physics.named.model.jnt_range[JOINTS]
        limited = self.physics.named.model.jnt_limited[JOINTS].astype(bool)
        
        # Generate random joint positions within limits
        qpos = np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            if limited[i]:
                low, high = joint_ranges[i]
                qpos[i] = np.random.uniform(low, high)
            else:
                qpos[i] = np.random.uniform(0, 2 * np.pi)
        
        # Set joint positions and update viewer
        self.physics.named.data.qpos[JOINTS] = qpos
        self.physics.forward()
        if self.viewer.is_running():
            self.viewer.sync()
    
    def solve_ik(self, target_pos, target_quat=None):
        """Solve inverse kinematics for the given target position and orientation."""
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name=SITE_NAME,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=JOINTS,
            tol=1e-6,
            max_steps=300,
            inplace=False
        )
        return result
    
    def animate_to_target(self, target_qpos):
        """Animate the arm from current position to target position."""
        start_qpos = self.physics.named.data.qpos[JOINTS].copy()
        
        for step in range(ANIMATION_STEPS + 1):
            # Interpolate between start and target positions
            alpha = step / ANIMATION_STEPS
            current_qpos = (1 - alpha) * start_qpos + alpha * target_qpos
            
            # Set the joint positions
            self.physics.named.data.qpos[JOINTS] = current_qpos
            self.physics.forward()
            
            # Update viewer
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.02)  # 50 FPS
            else:
                break
    
    def update_target_pose(self, new_pos, new_quat=None):
        """Update the target position and orientation."""
        self.target_pos = new_pos.copy()
        if new_quat is not None:
            self.target_quat = new_quat.copy()
        
        # Update the target body in the simulation
        self.physics.named.model.body_pos['target'] = self.target_pos
        if new_quat is not None:
            self.physics.named.model.body_quat['target'] = self.target_quat
        
        self.physics.forward()
        if self.viewer.is_running():
            self.viewer.sync()
    
    def update_target_position(self, new_pos):
        """Update just the target position (for backward compatibility)."""
        self.update_target_pose(new_pos)
    
    def run_demo(self):
        """Run the interactive demo."""
        print("IK Demo Controls:")
        print("- Press 'r' to randomize arm position")
        print("- Press 's' to solve IK and animate to target")
        print("- Press 't' to set new random target")
        print("- Press 'q' to quit")
        print()
        
        # Set initial random configuration
        self.set_random_arm_configuration()
        
        while self.viewer.is_running():
            # Get current end-effector position
            current_pos = self.physics.named.data.site_xpos[SITE_NAME].copy()
            distance = np.linalg.norm(current_pos - self.target_pos)
            
            print(f"\rEnd-effector distance to target: {distance:.4f}m", end="", flush=True)
            
            # Check for keyboard input (simplified - in real application you'd use proper input handling)
            key = input("\nEnter command (r/s/t/q): ").lower().strip()
            
            if key == 'r':
                print("Randomizing arm position...")
                self.set_random_arm_configuration()
                
            elif key == 's':
                print("Solving IK and animating to target...")
                result = self.solve_ik(self.target_pos, self.target_quat)
                
                if result.success:
                    print(f"IK solved in {result.steps} steps with error {result.err_norm:.6f}")
                    self.animate_to_target(result.qpos[self.physics.named.model.jnt_qposadr[JOINTS]])
                else:
                    print("IK failed to converge!")
                    
            elif key == 't':
                print("Setting new random target...")
                # Generate random target within reachable workspace
                new_target_pos = np.array([
                    np.random.uniform(-0.6, 0.6),
                    np.random.uniform(-0.6, 0.6),
                    np.random.uniform(0.1, 0.8)
                ])
                
                # Generate random orientation quaternion
                # Method: generate random unit quaternion by normalizing 4 random Gaussian numbers
                new_target_quat = np.random.randn(4)
                new_target_quat = new_target_quat / np.linalg.norm(new_target_quat)
                
                self.update_target_pose(new_target_pos, new_target_quat)
                print(f"New target pos: [{new_target_pos[0]:.3f}, {new_target_pos[1]:.3f}, {new_target_pos[2]:.3f}]")
                print(f"New target quat: [{new_target_quat[0]:.3f}, {new_target_quat[1]:.3f}, {new_target_quat[2]:.3f}, {new_target_quat[3]:.3f}]")
                
            elif key == 'q':
                break
                
        self.viewer.close()

def main():
    """Main function to run the demo."""
    print("Starting IK Demo...")
    demo = IKDemo()
    demo.run_demo()
    print("Demo completed.")

if __name__ == '__main__':
    main()