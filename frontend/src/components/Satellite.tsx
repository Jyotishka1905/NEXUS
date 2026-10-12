import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";

function SatelliteBody() {
  return (
    <group>
      {/* Main satellite body */}
      <mesh>
        <boxGeometry args={[0.28, 0.38, 0.28]} />
        <meshStandardMaterial
          color="#c7d0dc"
          metalness={0.8}
          roughness={0.25}
        />
      </mesh>

      {/* Left solar panel */}
      <mesh position={[-0.55, 0, 0]}>
        <boxGeometry args={[0.65, 0.03, 0.32]} />
        <meshStandardMaterial
          color="#0755a8"
          metalness={0.65}
          roughness={0.3}
        />
      </mesh>

      {/* Right solar panel */}
      <mesh position={[0.55, 0, 0]}>
        <boxGeometry args={[0.65, 0.03, 0.32]} />
        <meshStandardMaterial
          color="#0755a8"
          metalness={0.65}
          roughness={0.3}
        />
      </mesh>

      {/* Antenna */}
      <mesh position={[0, 0.28, 0]}>
        <cylinderGeometry args={[0.015, 0.015, 0.35, 12]} />

        <meshStandardMaterial
          color="#d9e2ec"
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Antenna dish */}
      <mesh position={[0, 0.45, 0]}>
        <sphereGeometry args={[0.07, 16, 16]} />

        <meshStandardMaterial
          color="#f2f5f8"
          metalness={0.7}
          roughness={0.2}
        />
      </mesh>

      {/* Blue scanner light */}
      <mesh position={[0, -0.22, 0]}>
        <sphereGeometry args={[0.045, 16, 16]} />

        <meshBasicMaterial color="#00aaff" />
      </mesh>
    </group>
  );
}

export default function Satellite() {
  const satelliteRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (!satelliteRef.current) return;

    const time = clock.getElapsedTime();

    const radius = 3.8;
    const speed = 0.22;

    const angle = time * speed;

    satelliteRef.current.position.x =
      Math.cos(angle) * radius;

    satelliteRef.current.position.z =
      Math.sin(angle) * radius;

    satelliteRef.current.position.y =
      Math.sin(angle * 1.8) * 0.45;

    // Face direction of movement
    satelliteRef.current.rotation.y = -angle;
  });

  return (
    <group ref={satelliteRef}>
      <SatelliteBody />
    </group>
  );
}