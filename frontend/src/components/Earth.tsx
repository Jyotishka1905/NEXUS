import { useFrame } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import { useRef } from "react";
import * as THREE from "three";

export default function Earth() {
  const earthRef = useRef<THREE.Mesh>(null);

  const textures = useTexture({
    map: "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg",
    normalMap: "https://threejs.org/examples/textures/planets/earth_normal_2048.jpg",
    specularMap:
      "https://threejs.org/examples/textures/planets/earth_specular_2048.jpg",
  });

  useFrame((_, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.08;
    }
  });

  return (
    <group>
      {/* Earth */}
      <mesh ref={earthRef}>
        <sphereGeometry args={[2.35, 96, 96]} />

        <meshPhongMaterial
          map={textures.map}
          normalMap={textures.normalMap}
          specularMap={textures.specularMap}
          specular={new THREE.Color("#333333")}
          shininess={15}
        />
      </mesh>

      {/* Atmospheric glow */}
      <mesh scale={1.035}>
        <sphereGeometry args={[2.35, 96, 96]} />

        <meshBasicMaterial
          color="#1688ff"
          transparent
          opacity={0.12}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Outer atmosphere */}
      <mesh scale={1.07}>
        <sphereGeometry args={[2.35, 64, 64]} />

        <meshBasicMaterial
          color="#1688ff"
          transparent
          opacity={0.045}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
}