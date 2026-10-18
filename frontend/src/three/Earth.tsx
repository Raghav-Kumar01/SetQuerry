import { Canvas, useFrame } from "@react-three/fiber";
import { useRef } from "react";
import type { Mesh } from "three";

function Globe() {
  const globe = useRef<Mesh>(null);
  useFrame((_, delta) => {
    if (globe.current) globe.current.rotation.y += delta * 0.12;
  });
  return (
    <mesh ref={globe}>
      <sphereGeometry args={[1.45, 48, 48]} />
      <meshStandardMaterial color="#1597bb" roughness={0.55} metalness={0.2} />
    </mesh>
  );
}

export function Earth() {
  return (
    <div className="earth-canvas" aria-label="Rotating 3D Earth visualization">
      <Canvas camera={{ position: [0, 0, 4.3], fov: 45 }} dpr={[1, 1.5]}>
        <color attach="background" args={["#081120"]} />
        <ambientLight intensity={0.45} />
        <directionalLight position={[4, 2, 3]} intensity={2.2} color="#aeeaff" />
        <pointLight position={[-3, -2, 2]} intensity={1.1} color="#2f70ff" />
        <Globe />
      </Canvas>
    </div>
  );
}
