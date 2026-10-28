// Pochi D435 v4 continuous wedge. Units: mm. X forward, Y left, Z up.
// The print STL is rotated +90 about X: original +Y is the build direction.
angle = 30;
print_pose = true;
width = 50;
// M3 heat-set insert OD 4.6, L 5.7. Full round bore is printed directly.
insert_hole = 4.0;
insert_depth = 6.7;
$fn = 96;
assert(angle >= 25 && angle <= 45);
function n() = [cos(angle),0,-sin(angle)];
function v() = [sin(angle),0,cos(angle)];
function c() = [12+45*sin(angle),0,2];
module cyl_x(r,h) { rotate([0,90,0]) cylinder(r=r,h=h,center=true); }
module teardrop_x(r,h) {
  ty=r/2;
  tz=r*sqrt(3)/2;
  union() {
    cyl_x(r,h);
    rotate([0,90,0]) linear_extrude(height=h,center=true)
      polygon([[-tz,ty],[tz,ty],[0,2*r]]);
  }
}
module camera_frame() { translate(c()) rotate([0,angle,0]) children(); }
module wedge() {
  hull() for(y=[-width/2,width/2]) {
    for(x=[0,8],z=[-32,32]) translate([x,y,z]) cube(.0001,center=true);
    for(s=[-37,37],d=[-8,0]) {
      p=c()+v()*s+n()*d;
      translate([p[0],y,p[2]]) cube(.0001,center=true);
    }
  }
}
module bracket() {
  difference() {
    wedge();
    for(y=[-15,15],z=[-24,24]) {
      translate([insert_depth/2-.01,y,z]) cyl_x(insert_hole/2,insert_depth+.02);
    }
    for(s=[-22.5,22.5]) camera_frame() {
      translate([-3,0,s]) teardrop_x(1.7,10);
      translate([-78,0,s]) teardrop_x(3.5,144);
    }
  }
}
if(print_pose) translate([0,40,25]) rotate([90,0,0]) bracket();
else bracket();
