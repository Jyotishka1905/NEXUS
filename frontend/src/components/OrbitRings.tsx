import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";

export default function OrbitRings() {
  const ring1 = useRef<THREE.Mesh>(null);
  const ring2 = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (ring1.current) {
      ring1.current.rotation.z += delta * 0.05;
      ring1.current.rotation.y += delta * 0.03;
    }

    if (ring2.current) {
      ring2.current.rotation.z -= delta * 0.035;
      ring2.current.rotation.x += delta * 0.02;
    }
  });

  return (
    <group>
      <mesh
        ref={ring1}
        rotation={[Math.PI / 2.5, 0, 0]}
      >
        <torusGeometry args={[3.05, 0.008, 8, 256]} />

        <meshBasicMaterial
          color="#1688ff"
          transparent
          opacity={0.65}
        />
      </mesh>

      <mesh
        ref={ring2}
        rotation={[Math.PI / 2.1, 0.5, 0]}
      >
        <torusGeometry args={[3.55, 0.006, 8, 256]} />

        <meshBasicMaterial
          color="#42baff"
          transparent
          opacity={0.35}
        />
      </mesh>
    </group>
  );
}