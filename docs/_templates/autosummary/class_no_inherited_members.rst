{#
   This is very similar to the default class template, except this one is used
   when we don't want to generate any inherited methods. Since we don't generate
   any inherited methods, we include a show-inheritance call to giver users an
   indication there may be inherited class members. Additionally, we also prevent
   duplication of the __init__ method by removing it from the automethod
   call in the default.
-#}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
