import{l as p,b as E,R as t,C as u,s as f,W as y,r as $,L as S,_ as I,u as L,o as w,P as N}from"./index-DVp1b9Fj.js";function k(e){return p.pick(e,["className","id"])}const P="text__single-line",v=e=>{const a=k(e),n=E(e.className,P),s={maxWidth:e.maxWidth?e.maxWidth:"",alignSelf:e.alignSelf?e.alignSelf:""},r=Array.isArray(e.children)?e.children.join(""):e.children,i={className:n,style:s,title:r},l=e.as||"span";return t.createElement(l,{...a,...i},e.children)},W={small:u`
    font-size: 12px;
  `,medium:u`
    font-size: 14px;
  `,large:u`
    font-size: 16px;
  `},g=f(y)`
  ${e=>W[e.size]};
  margin: 4px 4px 4px 4px;
  ${e=>e.$pos==="left"?"margin-left: -4px;":"margin-right: -4px;"}
  display: flex;
  align-items: center;
  opacity: ${e=>e.$opacity};
  ${e=>e.$cursor?`cursor: ${e.$cursor};`:""}
`;var x=(e=>(e[e.TAG=0]="TAG",e[e.ALIAS=1]="ALIAS",e[e.PROTECTED_ALIAS=2]="PROTECTED_ALIAS",e))(x||{});function b(e){switch(e){case"tag":return 0;case"alias":return 1;case"protected-alias":return 2;default:return 0}}function z(e,a){if(!e)return"tag-lightGray";switch(a){case 0:return"tag-teal-light";case 1:return"tag-sienna-light";case 2:return"tag-purple";default:return"tag-lightGray"}}const h=t.memo(({size:e,tag:a,noun:n,canDelete:s,showColor:r,onDelete:i,onClick:l})=>{const[c,m]=$.useState(!1),o=e||"large";n=n??"tag",s=s??!0,r=r??!0;const A=z(r,b(n));return t.createElement(S,{style:{marginLeft:"2px",maxWidth:"220px"},className:c?`run-tag ${o} tag-red-alert`:`run-tag ${o} ${A}`,key:a.name,onClick:l},t.createElement(g,{name:n==="tag"?"tag-latest":"email-at",size:o,$pos:"left"}),t.createElement(v,{alignSelf:"center"},a.name),s&&i&&t.createElement(g,{className:"delete-tag",name:"close-latest",size:o,onClick:d=>{d.stopPropagation(),i(d)},onMouseOver:()=>m(!0),onMouseOut:()=>m(!1),$pos:"right",$opacity:.6,$cursor:"pointer"}))});t.memo(({size:e,tags:a,enableDelete:n,noun:s,deleteTag:r,onClick:i})=>t.createElement("span",{className:"run-tags"},I.sortBy(a,"name").map(l=>t.createElement(h,{key:l.name,tag:l,size:e,onDelete:n&&r?c=>{r&&(c.stopPropagation(),r(l))}:void 0,onClick:()=>i?.(l.name),noun:s}))));const C=f.div`
  width: 100%;
  height: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  margin: auto;
  text-align: center;
  wordbreak: normal;
  display: flex;
  flex-direction: row;
  align-content: space-around;
  justify-content: left;
  align-items: center;
  -ms-overflow-style: none; /* IE and Edge */
  scrollbar-width: none; /* Firefox */
  &::-webkit-scrollbar {
    display: none;
  }
`,G=e=>{const a=L(w({artifactAlias:e.input}));return a.loading?t.createElement(N,null):a.result==null?t.createElement("div",null,"-"):t.createElement(C,null,a.result.map(n=>t.createElement(h,{key:n,tag:{name:n,colorIndex:x.ALIAS},noun:"alias"})))};export{G as default};
//# sourceMappingURL=Component-DY3AlQCe.js.map
