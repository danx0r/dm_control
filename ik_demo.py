#!/usr/bin/env python3
"""Demo script showing inverse kinematics with robotic arm visualization."""

import argparse
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
    def __init__(self, model_name, inplace=False):
        # Target position and orientation
        self.target_pos = np.array([.15, -.1, .1])
        # Generate random initial orientation
        # self.target_quat = np.random.randn(4)
        self.target_quat = [0, 0, 1, 0]
        
        # Store inplace parameter for IK solver
        self.inplace = inplace
        
        # Load the model directly from MJCF file
        arm_xml = assets.get_contents(model_name).decode('utf-8')
        self.physics = mujoco.Physics.from_xml_string(arm_xml)
        
        # Calculate workspace bounds based on arm geometry
        self._calculate_workspace_bounds()
        
        # Set initial target pose
        self.physics.named.model.body_pos['target'] = self.target_pos
        self.physics.named.model.body_quat['target'] = self.target_quat
        self.physics.forward()
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
        
        # Position camera closer to the arm
        self._setup_camera()
    
    def _setup_camera(self):
        """Position the camera for a better view of the arm."""
        if self.viewer.is_running():
            # Get arm base position for reference
            base_pos = self._get_base_position()
            
            # Set camera position closer to the arm
            # Camera distance (closer to arm)
            self.viewer.cam.distance = 0.75  # Closer than default
            
            # Camera elevation angle (looking down slightly)
            self.viewer.cam.elevation = -10  # degrees
            
            # Camera azimuth (side angle)
            self.viewer.cam.azimuth = 45  # degrees
            
            # Camera target point (center on arm base)
            self.viewer.cam.lookat[0] = base_pos[0]
            self.viewer.cam.lookat[1] = base_pos[1] 
            self.viewer.cam.lookat[2] = base_pos[2] + 0.3  # Slightly above base
        
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
    
    def _calculate_workspace_bounds(self):
        """Calculate workspace bounds by analyzing the model's kinematic chain."""
        # Get the kinematic chain from base to end-effector site
        site_id = self.physics.model.name2id(SITE_NAME, 'site')
        site_body_id = self.physics.model.site_bodyid[site_id]
        
        # Calculate maximum reach by analyzing joint limits and link lengths
        max_reach = self._calculate_max_reach()
        min_reach = self._calculate_min_reach()
        
        # Get base position (assuming arm is attached to world or first body)
        base_pos = self._get_base_position()
        
        # Calculate conservative workspace bounds
        safety_margin = 0.05  # 5cm safety margin
        
        self.workspace_bounds = {
            'x_range': (base_pos[0] - max_reach + safety_margin, 
                       base_pos[0] + max_reach - safety_margin),
            'y_range': (base_pos[1] - max_reach + safety_margin, 
                       base_pos[1] + max_reach - safety_margin),
            'z_range': (max(0.1, base_pos[2] + min_reach), 
                       base_pos[2] + max_reach - safety_margin),
            'max_reach': max_reach - safety_margin,
            'min_reach': min_reach,
            'base_pos': base_pos
        }
        
        print(f"Calculated workspace bounds:")
        print(f"  Max reach: {max_reach:.3f}m")
        print(f"  Min reach: {min_reach:.3f}m") 
        print(f"  Base position: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
    
    def _get_base_position(self):
        """Get the base position of the arm."""
        # Find the root body of the kinematic chain
        # Typically this is the first joint's parent body
        if len(JOINTS) > 0:
            joint_id = self.physics.model.name2id(JOINTS[0], 'joint')
            body_id = self.physics.model.jnt_bodyid[joint_id]
            body_name = self.physics.model.id2name(body_id, 'body')
            return self.physics.named.model.body_pos[body_name].copy()
        return np.zeros(3)
    
    def _calculate_max_reach(self):
        """Calculate maximum reach by analyzing the kinematic chain."""
        # Method 1: Sum of link lengths (conservative upper bound)
        total_link_length = 0.0
        
        # Get all bodies in the kinematic chain
        site_id = self.physics.model.name2id(SITE_NAME, 'site')
        site_body_id = self.physics.model.site_bodyid[site_id]
        
        # Traverse the kinematic chain and sum distances between joints
        for i, joint_name in enumerate(JOINTS):
            joint_id = self.physics.model.name2id(joint_name, 'joint')
            body_id = self.physics.model.jnt_bodyid[joint_id]
            
            if i < len(JOINTS) - 1:
                # Distance to next joint
                next_joint_id = self.physics.model.name2id(JOINTS[i + 1], 'joint')
                next_body_id = self.physics.model.jnt_bodyid[next_joint_id]
                
                # Get relative positions
                body_name = self.physics.model.id2name(body_id, 'body')
                next_body_name = self.physics.model.id2name(next_body_id, 'body')
                body_pos = self.physics.named.model.body_pos[body_name]
                next_body_pos = self.physics.named.model.body_pos[next_body_name]
                
                link_length = np.linalg.norm(next_body_pos - body_pos)
                total_link_length += link_length
            else:
                # Distance from last joint to end-effector site
                body_name = self.physics.model.id2name(body_id, 'body')
                body_pos = self.physics.named.model.body_pos[body_name]
                site_pos = self.physics.named.model.site_pos[SITE_NAME]
                
                # Site position is relative to its body, so add the offset
                final_link_length = np.linalg.norm(site_pos)
                total_link_length += final_link_length
        
        # Method 2: Empirical measurement by setting extreme joint positions
        empirical_max = self._measure_empirical_reach()
        
        # Use the more conservative (smaller) of the two estimates
        calculated_max = min(total_link_length, empirical_max)
        
        print(f"Link length sum: {total_link_length:.3f}m")
        print(f"Empirical max reach: {empirical_max:.3f}m")
        
        return calculated_max
    
    def _measure_empirical_reach(self):
        """Measure maximum reach empirically by testing joint configurations."""
        # Store current joint positions
        original_qpos = self.physics.named.data.qpos[JOINTS].copy()
        
        max_distance = 0.0
        base_pos = self._get_base_position()
        
        # Sample various joint configurations to find maximum reach
        n_samples = 100
        
        for _ in range(n_samples):
            # Generate random joint configuration within limits
            joint_ranges = self.physics.named.model.jnt_range[JOINTS]
            limited = self.physics.named.model.jnt_limited[JOINTS].astype(bool)
            
            qpos = np.zeros(len(JOINTS))
            for i, joint in enumerate(JOINTS):
                if limited[i]:
                    low, high = joint_ranges[i]
                    qpos[i] = np.random.uniform(low, high)
                else:
                    qpos[i] = np.random.uniform(-np.pi, np.pi)
            
            # Set configuration and measure end-effector position
            self.physics.named.data.qpos[JOINTS] = qpos
            self.physics.forward()
            
            ee_pos = self.physics.named.data.site_xpos[SITE_NAME]
            distance = np.linalg.norm(ee_pos - base_pos)
            max_distance = max(max_distance, distance)
        
        # Restore original joint positions
        self.physics.named.data.qpos[JOINTS] = original_qpos
        self.physics.forward()
        
        return max_distance
    
    def _calculate_min_reach(self):
        """Calculate minimum reach (when arm is maximally contracted)."""
        # Store current joint positions
        original_qpos = self.physics.named.data.qpos[JOINTS].copy()
        
        min_distance = float('inf')
        base_pos = self._get_base_position()
        
        # Try to find configuration that minimizes reach
        # This is often when joints are at extreme positions that fold the arm
        n_samples = 50
        
        for _ in range(n_samples):
            # Generate joint configuration biased toward folding the arm
            joint_ranges = self.physics.named.model.jnt_range[JOINTS]
            limited = self.physics.named.model.jnt_limited[JOINTS].astype(bool)
            
            qpos = np.zeros(len(JOINTS))
            for i, joint in enumerate(JOINTS):
                if limited[i]:
                    low, high = joint_ranges[i]
                    # Bias toward extreme values that might fold the arm
                    if np.random.random() < 0.5:
                        qpos[i] = low + 0.1 * (high - low)  # Near lower limit
                    else:
                        qpos[i] = high - 0.1 * (high - low)  # Near upper limit
                else:
                    qpos[i] = np.random.choice([-np.pi + 0.1, np.pi - 0.1])
            
            # Set configuration and measure end-effector position
            self.physics.named.data.qpos[JOINTS] = qpos
            self.physics.forward()
            
            ee_pos = self.physics.named.data.site_xpos[SITE_NAME]
            distance = np.linalg.norm(ee_pos - base_pos)
            min_distance = min(min_distance, distance)
        
        # Restore original joint positions
        self.physics.named.data.qpos[JOINTS] = original_qpos
        self.physics.forward()
        
        # Add small safety margin to minimum reach
        return max(0.05, min_distance * 0.9)
    
    def _generate_safe_target_position(self):
        """Generate a target position within the safe workspace."""
        attempts = 0
        max_attempts = 100
        base_pos = self.workspace_bounds['base_pos']
        
        while attempts < max_attempts:
            # Generate candidate position
            pos = np.array([
                np.random.uniform(*self.workspace_bounds['x_range']),
                np.random.uniform(*self.workspace_bounds['y_range']),
                np.random.uniform(*self.workspace_bounds['z_range'])
            ])
            
            # Check if position is within reach constraints
            distance_from_base = np.linalg.norm(pos - base_pos)
            if (self.workspace_bounds['min_reach'] <= distance_from_base <= 
                self.workspace_bounds['max_reach']):
                return pos
            
            attempts += 1
        
        # Fallback to a known safe position relative to base
        fallback_pos = base_pos + np.array([0.3, 0.2, 0.4])
        
        # Ensure fallback is within bounds
        distance = np.linalg.norm(fallback_pos - base_pos)
        if distance > self.workspace_bounds['max_reach']:
            # Scale down to fit within max reach
            direction = (fallback_pos - base_pos) / distance
            fallback_pos = base_pos + direction * (self.workspace_bounds['max_reach'] * 0.8)
        
        return fallback_pos
    
    def solve_ik(self, target_pos, target_quat=None, max_attempts=5):
        """Solve inverse kinematics with multiple attempts from different initial configurations."""
        best_result = None
        best_error = float('inf')
        best_qpos = None
        
        # Separate tracking for actual geometric distance (for non-convergence cases)
        best_distance = float('inf')
        best_distance_qpos = None
        best_distance_result = None
        
        # Store current joint positions
        original_qpos = self.physics.named.data.qpos[JOINTS].copy()
        
        for attempt in range(max_attempts):
            if attempt > 0:
                # Randomize initial joint configuration for subsequent attempts
                self.set_random_arm_configuration()
            
            # Attempt IK solve
            result = ik.qpos_from_site_pose(
                physics=self.physics,
                site_name=SITE_NAME,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=JOINTS,
                tol=1e-6,
                max_steps=500,  # Increased from 300
                inplace=self.inplace
            )
            
            # Calculate actual end-effector distance for this attempt
            actual_ee_pos = self.physics.named.data.site_xpos[SITE_NAME]
            actual_distance = np.linalg.norm(actual_ee_pos - target_pos)
            
            if result.success:
                # Restore original position and return successful result
                self.physics.named.data.qpos[JOINTS] = original_qpos
                self.physics.forward()
                return result
            else:
                # Track best IK error (for potential convergence in future attempts)
                if result.err_norm < best_error:
                    best_result = result
                    best_error = result.err_norm
                    best_qpos = self.physics.named.data.qpos[JOINTS].copy()
                
                # Always track best actual distance (for non-convergence cases)
                if actual_distance < best_distance:
                    best_distance = actual_distance
                    best_distance_qpos = self.physics.named.data.qpos[JOINTS].copy()
                    best_distance_result = result
                    print(f"  Attempt {attempt + 1}: IK error {result.err_norm:.6f}, actual distance {actual_distance:.6f}m (best distance so far)")
                else:
                    print(f"  Attempt {attempt + 1}: IK error {result.err_norm:.6f}, actual distance {actual_distance:.6f}m")
        
        # For non-convergence, prioritize best actual distance over best IK error
        if best_distance_result is not None and best_distance_qpos is not None:
            # Use the configuration with the best actual distance
            final_result = best_distance_result
            final_qpos = best_distance_qpos
            
            # Update result with the best distance joint configuration
            final_result.qpos[self.physics.named.model.jnt_qposadr[JOINTS]] = final_qpos
            
            # Verify the distance
            self.physics.named.data.qpos[JOINTS] = final_qpos
            self.physics.forward()
            actual_ee_pos = self.physics.named.data.site_xpos[SITE_NAME]
            verified_distance = np.linalg.norm(actual_ee_pos - target_pos)
            
            print(f"IK failed to converge. Returning configuration with best actual distance:")
            print(f"  Best IK error found: {best_error:.6f}")
            print(f"  Chosen actual distance: {verified_distance:.6f}m")
            print(f"  Chosen IK error: {final_result.err_norm:.6f}")
            
        # Restore original joint positions
        self.physics.named.data.qpos[JOINTS] = original_qpos
        self.physics.forward()
        
        # Return best distance attempt, fallback to best error, then last result
        if best_distance_result is not None:
            return best_distance_result
        elif best_result is not None:
            return best_result
        else:
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
                    # Only animate for successful convergence
                    self.animate_to_target(result.qpos[self.physics.named.model.jnt_qposadr[JOINTS]])
                else:
                    print(f"IK failed to converge! Best solution error: {result.err_norm:.6f}")
                    print("(No animation - try 'r' to randomize arm position and try again)")
                    
            elif key == 't':
                print("Setting new random target...")
                # Generate random target within safe reachable workspace
                new_target_pos = self._generate_safe_target_position()
                
                # Generate random orientation quaternion
                # Method: generate random unit quaternion by normalizing 4 random Gaussian numbers
                new_target_quat = np.random.randn(4)
                new_target_quat = new_target_quat / np.linalg.norm(new_target_quat)
                
                self.update_target_pose(new_target_pos, new_target_quat)
                distance_from_base = np.linalg.norm(new_target_pos)
                print(f"New target pos: [{new_target_pos[0]:.3f}, {new_target_pos[1]:.3f}, {new_target_pos[2]:.3f}] (dist: {distance_from_base:.3f}m)")
                print(f"New target quat: [{new_target_quat[0]:.3f}, {new_target_quat[1]:.3f}, {new_target_quat[2]:.3f}, {new_target_quat[3]:.3f}]")
                
            elif key == 'q':
                break
                
        self.viewer.close()

def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description='IK Demo with robotic arm visualization')
    parser.add_argument('model', help='MJCF model name (e.g., tycho_arm.mjcf)')
    parser.add_argument('--inplace', action='store_true', 
                       help='Use in-place IK solver (modifies physics state)')
    
    args = parser.parse_args()
    
    print(f"Starting IK Demo with model: {args.model}")
    print(f"In-place IK: {args.inplace}")
    
    demo = IKDemo(args.model, args.inplace)
    demo.run_demo()
    print("Demo completed.")

if __name__ == '__main__':
    main()