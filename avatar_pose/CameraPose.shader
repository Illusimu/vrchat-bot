Shader "DesktopBody/CameraPose" {
 SubShader {
  Tags { "Queue"="Overlay+100" "IgnoreProjector"="True" "DisableBatching"="True" }
  Pass {
   Cull Off ZWrite Off ZTest Always
   CGPROGRAM
   #pragma vertex vert
   #pragma fragment frag
   #pragma target 4.0
   #pragma multi_compile_instancing
   #include "UnityCG.cginc"
   float3 _VRChatScreenCameraPos, _VRChatPhotoCameraPos;
   float4 _VRChatScreenCameraRot, _VRChatPhotoCameraRot;
   float _VRChatCameraMode, _VRChatMirrorMode;
   struct Input { float4 vertex:POSITION; float2 uv:TEXCOORD0; UNITY_VERTEX_INPUT_INSTANCE_ID };
   struct Output { float4 pos:SV_POSITION; float2 uv:TEXCOORD0; UNITY_VERTEX_OUTPUT_STEREO };
   Output vert(Input v) {
    UNITY_SETUP_INSTANCE_ID(v);
    Output o;
    UNITY_INITIALIZE_OUTPUT(Output,o);
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    o.uv=v.uv;
    float2 screen=float2(.04,.06)+v.uv*float2(.20,.16);
    o.pos=float4(screen.x*2-1,1-screen.y*2,.5,1);
    return o;
   }
   uint word(uint i) {
    if(i==0) return 0xC35A76E1u;
    if(i<4) return asuint(_VRChatScreenCameraPos[i-1]);
    if(i<8) return asuint(_VRChatScreenCameraRot[i-4]);
    if(i<11) return asuint(_VRChatPhotoCameraPos[i-8]);
    if(i<15) return asuint(_VRChatPhotoCameraRot[i-11]);
    if(i<18) return asuint(_WorldSpaceCameraPos[i-15]);
    if(i<21) return asuint(UNITY_MATRIX_V[0][i-18]);
    if(i<24) return asuint(UNITY_MATRIX_V[1][i-21]);
    return asuint(-UNITY_MATRIX_V[2][i-24]);
   }
   float4 frag(Output p):SV_Target {
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(p);
    if (_VRChatCameraMode!=0 || _VRChatMirrorMode!=0) discard;
    uint row=min((uint)(p.uv.y*28),27u);
    uint bits;
    if(row==27) {
     bits=2166136261u;
     [unroll] for(uint i=0;i<27;i++) bits=(bits^word(i))*16777619u;
    } else bits=word(row);
    float bit=(bits>>min((uint)(p.uv.x*32),31u))&1u;
    return float4(bit,bit,bit,1);
   }
   ENDCG
  }
 }
}
