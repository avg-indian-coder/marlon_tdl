import xml.etree.ElementTree as ET

xml_data = '''
<additional>
    <e1Detector id="0.127_2.73_6_1__l0" lane="a104_0" pos="23" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.73_6_1__l1" lane="a104_1" pos="23" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.73_4_1__l0" lane="a15_0" pos="77.542050042" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.73_2_1__l0" lane="a103_0" pos="118" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.73_2_1__l1" lane="a103_1" pos="118" freq="1800" file="e1_output.xml"/>


    <e1Detector id="2.19_2.20_8_1__l0" lane="a72[0]_0" pos="32.8995678051" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_2.20_8_1__l1" lane="a72[0]_1" pos="32.8995678051" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_2.20_8_1__l2" lane="a72[0]_2" pos="32.8995678051" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.19_6_1__l0" lane="a11_0" pos="0" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.19_6_1__l1" lane="a11_1" pos="0" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.19_6_1__l2" lane="a11_2" pos="0" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_0.127_2_1__l0" lane="a67_0" pos="42" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_0.127_2_1__l1" lane="a67_1" pos="42" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_0.127_8_1__l0" lane="b46_0" pos="105" freq="1800" file="e1_output.xml"/>
<!--    <e1Detector id="2.19_0.127_8_1__l1" lane="b46_1" pos="105" freq="1800" file="e1_output.xml"/>  -->


    <e1Detector id="2.20_2.19_4_1__l0" lane="a17_0" pos="100" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.20_2.19_4_1__l1" lane="a17_1" pos="100" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.20_2_1__l0" lane="a18_0" pos="1" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.20_2_1__l1" lane="a18_1" pos="1" freq="1800" file="e1_output.xml"/>
<!--    <e1Detector id="0.127_2.20_2_1__l2" lane="a18_2" pos="1" freq="1800" file="e1_output.xml"/> -->
    <e1Detector id="2.20_2.21_8_1__l0" lane="a161_0" pos="20" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.20_2.21_8_1__l1" lane="a161_1" pos="20" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.20_2.21_8_1__l2" lane="a161_2" pos="20" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.20_6_1__l0" lane="a171_0" pos="166" freq="1800" file="e1_output.xml"/>


    <e1Detector id="2.9_2.20_6_1__l0" lane="a191_0" pos="12" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.9_2.10_8_1__l0" lane="a87[1][1]_0" pos="0" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.9_6_1__l0" lane="a153_0" pos="208.488866169" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.8_2.9_8_2__l0" lane="a189[1][1]_0" pos="142" freq="1800" file="e1_output.xml"/>


    <e1Detector id="2.6_2.10_6_1__l0" lane="a210_0" pos="10.6430697599" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.6_2.10_6_1__l1" lane="a210_1" pos="10.6430697599" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.6_2.10_6_1__l2" lane="a210_2" pos="10.6430697599" freq="1800" file="e1_output.xml"/>


    <e1Detector id="0.127_2.10_8_1__l0" lane="a117_0" pos="105" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_2.6_6_1__l0" lane="a113_0" pos="23.5161227555" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_2.6_6_1__l1" lane="a113_1" pos="23.5161227555" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_2.6_6_1__l2" lane="a113_2" pos="23.5161227555" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_2.21_6_1__l0" lane="a201_0" pos="11.5139549672" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_2.21_6_1__l1" lane="a201_1" pos="11.5139549672" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_2.21_6_1__l2" lane="a201_2" pos="11.5139549672" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_0.127_8_1__l0" lane="a134b_0" pos="145" freq="1800" file="e1_output.xml"/>
<!-->    <e1Detector id="2.10_0.127_8_1__l1" lane="a134b_1" pos="145" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.10_0.127_8_1__l2" lane="a134b_2" pos="145" freq="1800" file="e1_output.xml"/>   -->


    <e1Detector id="2.21_2.10_6_1__l0" lane="a202_0" pos="75" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.21_2.10_6_1__l1" lane="a202_1" pos="75" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.21_2.10_6_1__l2" lane="a202_2" pos="75" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.21_2.35_8_1__l0" lane="a204a[0]_0" pos="7.20261067361" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.21_2.35_8_1__l1" lane="a204a[0]_1" pos="7.20261067361" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.21_2.35_8_1__l2" lane="a204a[0]_2" pos="7.20261067361" freq="1800" file="e1_output.xml"/>


    <e1Detector id="2.35_2.38_8_1__l0" lane="a204[1][1]_0" pos="36" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_2.38_8_1__l1" lane="a204[1][1]_1" pos="36" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_2.38_8_1__l2" lane="a204[1][1]_2" pos="36" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_1.9_2_1__l0" lane="a54_0" pos="11.6007252321" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_1.9_2_1__l1" lane="a54_1" pos="11.6007252321" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_2.21_6_1__l0" lane="a203[1]_0" pos="220" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_2.21_6_1__l1" lane="a203[1]_1" pos="220" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_2.21_6_1__l2" lane="a203[1]_2" pos="220" freq="1800" file="e1_output.xml"/>
    <!--
    <e1Detector id="San_Felice.1__l0" lane="a203[1]_0" pos="11.9973887785" freq="1800" file="e1_output.xml"/>
    <e1Detector id="San_Felice.1__l1" lane="a203[1]_1" pos="11.9973887785" freq="1800" file="e1_output.xml"/>
    <e1Detector id="San_Felice.1__l2" lane="a203[1]_2" pos="11.9973887785" freq="1800" file="e1_output.xml"/>
    -->
    <e1Detector id="2.35_2.34_4_1__l0" lane="b12_0" pos="11.3754906176" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_2.34_4_1__l1" lane="b12_1" pos="11.3754906176" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.35_2.34_4_1__l2" lane="b12_2" pos="11.3754906176" freq="1800" file="e1_output.xml"/>


    <e1Detector id="0.127_2.33_2_1__l0" lane="b51_0" pos="42" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.33_2_1__l1" lane="b51_1" pos="42" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.33_0.127_6_1__l0" lane="b53_0" pos="18" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.33_2.34_8_4__l0" lane="b35[1][1][1][1]_0" pos="0" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.33_6_1__l0" lane="b55_0" pos="73" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.33_0.127_2_1__l0" lane="b50[0]_0" pos="19" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.33_2.32_4_1__l0" lane="b56[0]_0" pos="10." freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.33_2.32_4_1__l1" lane="b56[0]_1" pos="10." freq="1800" file="e1_output.xml"/>

    <e1Detector id="2.31_2.32_8_1__l0" lane="b101_0" pos="14.7630542185" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.31_1_1__l0" lane="b71_0" pos="0" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.31_2_1__l0" lane="b65_0" pos="162" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.31_6_1__l0" lane="b31_0" pos="153" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.82_2.31_6_1__l0" lane="b17[1]_0" pos="160" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.31_2.30_4_1__l0" lane="b8_0" pos="1.7" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.31_2.30_4_1__l1" lane="b8_1" pos="1.7" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.31_2.30_4_1__l2" lane="b8_2" pos="1.7" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.31_2.30_4_1__l3" lane="b8_3" pos="1.7" freq="1800" file="e1_output.xml"/>

    <e1Detector id="2.30_0.127_6_1__l0" lane="b4[1][1][1]_0" pos="36.7357969861" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.30_0.127_6_1__l1" lane="b4[1][1][1]_1" pos="36.7357969861" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.30_2_1__l0" lane="b3[0]_0" pos="137.4601581707" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.30_2_1__l1" lane="b3[0]_1" pos="137.4601581707" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.30_6_1__l0" lane="b4[1][1][0]_0" pos="143" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.30_6_1__l1" lane="b4[1][1][0]_1" pos="143" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.30_2.29_4_1__l0" lane="b10_0" pos="30" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.30_2.31_8_1__l0" lane="b9_0" pos="56.8078174024" freq="1800" file="e1_output.xml"/>

    <e1Detector id="2.31_2.82_2_1__l0" lane="b16[1]_0" pos="196" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_2.82_4_1__l0" lane="b26.-170_0" pos="150" freq="1800" file="e1_output.xml"/>
<!--    <e1Detector id="2.19_2.82_4_1__l1" lane="b26_1" pos="20" freq="1800" file="e1_output.xml"/>   -->
    <e1Detector id="2.82_2.18_2_2__l0" lane="b18_0" pos="15.9248731148" freq="1800" file="e1_output.xml"/>
<!--    <e1Detector id="2.82_2.18_2_2__l1" lane="b18_1" pos="15.9248731148" freq="1800" file="e1_output.xml"/>  -->
    <e1Detector id="0.127_2.82_8_1__l0" lane="b14_0" pos="295" freq="1800" file="e1_output.xml"/>


    <e1Detector id="2.19_2.32_6_1__l0" lane="b38[1][0]_0" pos="9.72072259695" freq="1800" file="e1_output.xml"/>
    <!-- e1Detector id="2.19_2.19_2_1__l0" lane="b39[1][1][1]_0" pos="24.3955036857" freq="1800" file="e1_output.xml"/ -->
    <e1Detector id="2.82_2.19_8_1__l0" lane="b36_0" pos="156" freq="1800" file="e1_output.xml"/>

<!--
    <e1Detector id="2.19_2.19_6_1__l0" lane="b45_0" pos="10" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_2.19_6_1__l1" lane="b45_1" pos="10" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_2.19_6_1__l2" lane="b45_2" pos="10" freq="1800" file="e1_output.xml"/>
-->
    <e1Detector id="2.19_2.18_4_1__l0" lane="b2[0]_0" pos="31.3626881232" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_2.18_4_1__l1" lane="b2[0]_1" pos="31.3626881232" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.19_2.18_4_1__l2" lane="b2[0]_2" pos="31.3626881232" freq="1800" file="e1_output.xml"/>


    <e1Detector id="2.82_2.18_2_1__l0" lane="b18_0" pos="140" freq="1800" file="e1_output.xml"/>
<!--    <e1Detector id="2.82_2.18_2_1__l1" lane="b18_1" pos="140" freq="1800" file="e1_output.xml"/>  -->
    <e1Detector id="2.18_2.82_6_1__l0" lane="b23_0" pos="18" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.18_2.82_6_1__l1" lane="b23_1" pos="18" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.18_2.19_8_1__l0" lane="b1[1]_0" pos="80" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.18_2.19_8_1__l1" lane="b1[1]_1" pos="80" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.18_2.19_8_1__l2" lane="b1[1]_2" pos="80" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.18_6_1__l0" lane="b22[1]_0" pos="14" freq="1800" file="e1_output.xml"/>
    <e1Detector id="0.127_2.18_6_1__l1" lane="b22[1]_1" pos="14" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.17_2.18_8_1__l0" lane="b1[0]_0" pos="329" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.17_2.18_8_1__l1" lane="b1[0]_1" pos="329" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.17_2.18_8_1__l2" lane="b1[0]_2" pos="329" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.18_2.17_4_2__l0" lane="b2[1][1][1]b_0" pos="55" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.18_2.17_4_2__l1" lane="b2[1][1][1]b_1" pos="55" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.18_2.17_4_2__l2" lane="b2[1][1][1]b_2" pos="55" freq="1800" file="e1_output.xml"/>

    <e1Detector id="2.32_2.31_4_1__l0" lane="b56[1][0]_0" pos="30" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.32_2.31_4_1__l1" lane="b56[1][0]_1" pos="30" freq="1800" file="e1_output.xml"/>

    <e1Detector id="2.30_0.127_2_1__l0" lane="b3[1]_0" pos="40" freq="1800" file="e1_output.xml"/>
    <e1Detector id="2.30_0.127_2_1__l1" lane="b3[1]_1" pos="40" freq="1800" file="e1_output.xml"/>


</additional>

'''

# Parse the XML data
root = ET.fromstring(xml_data)

# Iterate through each e1Detector element
for e1Detector in root.findall('e1Detector'):
    # Get the value of the 'lane' attribute
    lane_value = e1Detector.get('lane')
    # Set the 'id' attribute to the 'lane' value
    e1Detector.set('id', lane_value)

# Convert the modified XML back to a string
modified_xml_data = ET.tostring(root, encoding='utf-8').decode()

print(modified_xml_data)
